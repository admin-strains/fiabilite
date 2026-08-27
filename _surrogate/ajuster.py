r"""Les ajustements de metamodeles : PCE, krigeage, GEK, et le vecteur
gradient-augmente qui nourrit le GEPCK.

Ces cinq fonctions etaient definies dans le bloc `__main__` des deux scripts
d'etude. Elles ne dependaient de l'etude que par six valeurs -- la loi jointe,
`q`, `max_degree`, `n_var`, et deux interrupteurs -- qui sont ici des
arguments.

CE QUE L'EXTRACTION A REVELE
-----------------------------
1. **Une divergence silencieuse entre les deux etudes.** `build_metamodel_KRG`
   bornait l'optimisation des longueurs de correlation a `[1, 100]` sur la
   flexion pure et a `[0, 100]` sur le Moulin Blanc -- deux fichiers par
   ailleurs identiques au caractere pres. Une borne inferieure nulle autorise
   une longueur de correlation degeneree, c'est-a-dire un krigeage qui
   n'interpole plus rien. Aucune trace, aucun commentaire, aucun test ne
   signalait l'ecart. C'est la quatrieme divergence trouvee entre les deux
   copies. La borne est desormais un ARGUMENT : chaque etude garde sa valeur,
   mais elle est ecrite noir sur blanc a l'appel.

2. **Un `if` dont les deux branches etaient identiques.** Dans
   `build_metamodel_GEK`, `if do_GEK: sm = GEKPLS(...) else: sm = GEKPLS(...)`
   construisait exactement le meme objet, au caractere pres, dans les deux
   branches.

3. **Deux parametres jamais lus.** `calculate_PCE(xt, y_hf, all_grad_hf,
   metamodel_PCE)` n'utilisait ni `y_hf` ni `all_grad_hf`. Un lecteur pouvait
   croire que la composante PCE des gradients etait comparee aux gradients
   haute fidelite : elle ne l'est pas.

4. **Encore une hypothese `n_var == 2`**, dans l'etiquetage des termes du
   chaos polynomial (`mi[0]`, `mi[1]`, « u1 », « u2 »). Generalisee ; a deux
   variables l'etiquette produite est inchangee.
"""

import numpy as np
import openturns as ot

from smt.surrogate_models import GEKPLS


def _ecrire(message):
    print(message, flush=True)


def y_augmente(yt, all_grad):
    """Vecteur gradient-augmente y_dot (eq. 6, Zuhal et al.).

        y_dot = [y^1..y^n, dg/du1^1..dg/du1^n, ..., dg/dum^1..dg/dum^n]^T

    de longueur n * (1 + n_var).
    """
    y_flat = yt.flatten()
    grad_blocks = [all_grad[:, j] for j in range(all_grad.shape[1])]
    return np.concatenate([y_flat] + grad_blocks)


def _etiquette_terme(multi_indice):
    """« H2(u1)*H1(u3) » pour un multi-indice de degres.

    L'original ne lisait que `mi[0]` et `mi[1]`, et ecrivait « u1 »/« u2 » en
    dur : au-dela de deux variables l'etiquette etait fausse sans le dire.
    """
    morceaux = ["H%d(u%d)" % (int(d), i + 1)
                for i, d in enumerate(multi_indice) if int(d) != 0]
    return "*".join(morceaux) if morceaux else "1"


def ajuster_PCE(xt, y_hf, dist_X, q, max_degree, tracer=_ecrire):
    """Chaos polynomial creux : base hyperbolique anisotrope, selection LARS
    validee par leave-one-out corrige.

    `q < 1` resserre la base sur les termes de faible interaction ; c'est ce
    qui rend le degre eleve abordable avec peu de points. LARS gere P > N,
    d'ou l'absence de garde sur `max_degree`.
    """
    inputSample = ot.Sample(xt)
    outputSample = ot.Sample(y_hf)
    dist_U = dist_X.getStandardDistribution()

    n_var = inputSample.getDimension()
    enumerateFunction = ot.HyperbolicAnisotropicEnumerateFunction(n_var, q)
    basis = ot.OrthogonalProductPolynomialFactory(
        [ot.HermiteFactory()] * n_var, enumerateFunction)
    basis_size = enumerateFunction.getBasisSizeFromTotalDegree(max_degree)
    basisStrategy = ot.FixedStrategy(basis, basis_size)

    selectionStrategy = ot.LeastSquaresMetaModelSelectionFactory(
        ot.LARS(), ot.CorrectedLeaveOneOut())
    projectionStrategy = ot.LeastSquaresStrategy(selectionStrategy)

    algo = ot.FunctionalChaosAlgorithm(inputSample, outputSample, dist_U,
                                       basisStrategy, projectionStrategy)
    algo.run()
    result = algo.getResult()
    n_active = result.getCoefficients().getSize()
    tracer("PCE construite : basis_size=%d, coefficients actifs LARS=%d"
           % (basis_size, n_active))
    indices = result.getIndices()
    coeffs = result.getCoefficients()
    terms = ["%+.4f*%s" % (coeffs[k, 0], _etiquette_terme(enumerateFunction(indices[k])))
             for k in range(indices.getSize())]
    tracer("  PCE termes : %s" % " ".join(terms))
    return result.getMetaModel()


def composante_PCE(xt, metamodele_PCE):
    """Valeur et gradient du chaos polynomial AUX POINTS DU PLAN.

    C'est la part que le krigeage residuel n'aura pas a representer :
    `yr = y_hf - y_PCE`, `grad_r = grad_hf - grad_PCE`.

    L'original recevait aussi `y_hf` et `all_grad_hf` et ne les lisait pas.
    """
    U_doe = ot.Sample(xt)
    y_PCE = np.array(metamodele_PCE(U_doe))
    n_var = U_doe.getDimension()
    n_pts = U_doe.getSize()
    all_grad_PCE = np.zeros((n_pts, n_var))
    for i in range(n_pts):
        grad_pce_u = metamodele_PCE.gradient(U_doe[i])
        for j in range(n_var):
            all_grad_PCE[i, j] = grad_pce_u[j, 0]
    return y_PCE, all_grad_PCE


def ajuster_KRG(xt, yt, borner_theta=False, theta_min=1.0, theta_max=100.0,
                tracer=_ecrire):
    """Krigeage ordinaire a noyau exponentiel carre, tendance constante.

    `borner_theta` contraint les longueurs de correlation a
    `[theta_min, theta_max]`.

    ATTENTION -- `theta_min` differait entre les deux etudes sans que rien ne
    le dise : 1,0 sur la flexion pure, 0,0 sur le Moulin Blanc. Une borne
    inferieure nulle laisse l'optimiseur degenerer vers une correlation quasi
    nulle, c'est-a-dire un krigeage qui interpole le bruit. La valeur reste
    celle de chaque etude, mais elle est desormais visible a l'appel.
    """
    n_var = xt.shape[1]
    basis = ot.ConstantBasisFactory(n_var).build()
    # pour une tendance lineaire : ot.LinearBasisFactory(n_var).build()
    covarianceModel = ot.SquaredExponential([1.0] * n_var)
    algo_KRG = ot.KrigingAlgorithm(xt, yt, covarianceModel, basis)
    if borner_theta:
        algo_KRG.setOptimizationBounds(
            ot.Interval([theta_min] * n_var, [theta_max] * n_var))
    algo_KRG.run()
    result = algo_KRG.getResult()
    cov_opt = result.getCovarianceModel()
    tracer("  KRG theta=%s  sigma=%s"
           % (list(cov_opt.getScale()), list(cov_opt.getAmplitude())))
    return result.getMetaModel(), result


def ajuster_GEK(xt, yt, all_grad, n_comp=2, theta0=1e-2,
                theta_bounds=(1.0, 5.0), tracer=_ecrire):
    """Krigeage augmente par les gradients, avec reduction PLS (smt.GEKPLS).

    Le domaine `xlimits` est elargi d'une unite de part et d'autre du plan :
    smt normalise les entrees sur cette boite, et un point d'enrichissement
    tombant exactement sur un bord y serait mal conditionne.

    L'original portait un `if do_GEK: ... else: ...` dont les deux branches
    construisaient le MEME objet, au caractere pres.
    """
    xlimits = np.column_stack([xt.min(axis=0) - 1, xt.max(axis=0) + 1])
    sm = GEKPLS(
        n_comp=n_comp,
        theta0=[theta0] * n_comp,
        theta_bounds=list(theta_bounds),
        corr="squar_exp",
        poly="constant",
        xlimits=xlimits,
        print_global=False,
    )
    sm.set_training_values(xt, yt)
    for j in range(all_grad.shape[1]):
        sm.set_training_derivatives(xt, all_grad[:, j].reshape(-1, 1), j)
    sm.train()
    tracer("  GEK theta=%s  sigma=%.6f"
           % (list(sm.optimal_theta),
              float(np.sqrt(sm.optimal_par['sigma2']))))
    return sm
