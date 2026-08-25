"""
branche2.py — Python word-for-word translation of UQLab B2 functions

Sources:
  - core/uq_process_option.m
  - modules/uq_model/builtin/uq_metamodel/PCK/uq_PCK_initialize.m

uq_copyObj is handled by copy.deepcopy (same semantics as MATLAB uq_copyObj for structs).
uq_getInput is replaced by a global_input argument passed into uq_PCK_initialize.
"""

import copy
import numpy as np


# ---------------------------------------------------------------------------
# MATLAB type names → Python types accepted
# ---------------------------------------------------------------------------
_MATLAB_TYPE_MAP = {
    'char':     str,
    'logical':  bool,
    'double':   (int, float, np.integer, np.floating, np.ndarray),
    'struct':   dict,
    'cell':     list,
    'uq_input': dict,   # UQLab's uq_input object is a struct → dict in Python
}


def _python_class(value):
    """
    Return the MATLAB-equivalent class name of a Python value.
    Mirrors MATLAB's class() function for the types used in PCK.
    """
    if isinstance(value, bool):      # must come before int (bool is subclass of int)
        return 'logical'
    if isinstance(value, str):
        return 'char'
    if isinstance(value, (int, float, np.integer, np.floating, np.ndarray)):
        return 'double'
    if isinstance(value, dict):
        return 'struct'
    if isinstance(value, list):
        return 'cell'
    return type(value).__name__


# ---------------------------------------------------------------------------
# uq_process_option
#   Faithful translation of core/uq_process_option.m
#   Signature: [opt, AllOptions] = uq_process_option(AllOptions, OptionName,
#                                    Default, AllowedClasses, EmptyAsMissing)
# ---------------------------------------------------------------------------
def uq_process_option(AllOptions, OptionName, Default=None,
                      AllowedClasses=None, EmptyAsMissing=False):
    """
    Parse one named option from AllOptions dict.

    Returns
    -------
    opt : dict
        Keys: Disabled, Missing, Invalid, Name, Value, Type, Default
    AllOptions : dict
        Input dict with the found key removed (matching MATLAB behaviour).
    """
    # --- initialise the returned struct (lines 5-8 in MATLAB) ---
    opt = {
        'Disabled': False,
        'Missing':  False,
        'Invalid':  False,
    }

    # lines 9-13: set Name / Value / Type / Default when Default was supplied
    # (nargin > 2 in MATLAB — here Default is always provided by caller)
    opt['Name']    = OptionName
    opt['Value']   = Default
    opt['Type']    = _python_class(Default) if Default is not None else 'unknown'
    opt['Default'] = Default

    # --- disabled options list (lines 18-29) ---
    DisabledOptions = ['']
    if any(o.lower() == OptionName.lower() for o in DisabledOptions):
        opt['Disabled'] = True
        AllOptions = {k: v for k, v in AllOptions.items()
                      if k.lower() != OptionName.lower()}
        return opt, AllOptions

    # --- case-insensitive key lookup (lines 31-35) ---
    AOnames   = list(AllOptions.keys())
    OptionFound = [name for name in AOnames if name.lower() == OptionName.lower()]

    # --- EmptyAsMissing (lines 37-42) ---
    if EmptyAsMissing and OptionFound:
        key = OptionFound[0]
        val = AllOptions[key]
        if val is None or (hasattr(val, '__len__') and len(val) == 0):
            OptionFound = []
            AllOptions = {k: v for k, v in AllOptions.items() if k != key}

    # --- option not found → return default (lines 48-54) ---
    if len(OptionFound) == 0:
        opt['Missing'] = True
        return opt, AllOptions

    # --- more than one match → warn, keep first, remove others (lines 56-79) ---
    if len(OptionFound) > 1:
        print(f'\nWarning: There is more than one field referring to the option '
              f'"{OptionName}".')
        print(f'Only the value provided with name "{OptionFound[0]}" will be used.')
        for extra in OptionFound[1:]:
            AllOptions = {k: v for k, v in AllOptions.items() if k != extra}

    FoundName  = OptionFound[0]
    FoundClass = _python_class(AllOptions[FoundName])

    # --- type check (lines 90-115) ---
    if AllowedClasses is not None:
        if isinstance(AllowedClasses, str):
            AllowedClasses = [AllowedClasses]

        # build the union of accepted Python types
        accepted_py_types = tuple(
            _MATLAB_TYPE_MAP.get(c, type(None)) for c in AllowedClasses
        )

        if not isinstance(AllOptions[FoundName], accepted_py_types):
            allowed_list = ', '.join(AllowedClasses)
            print(f'\nWarning: The option provided "{FoundName}" is of type '
                  f'{FoundClass}, but the accepted types are: {allowed_list}')
            print(f'"{OptionName}" is set to its default value:')
            print(Default)
            opt['Name']    = FoundName
            opt['Invalid'] = True
            AllOptions     = {k: v for k, v in AllOptions.items() if k != FoundName}
            return opt, AllOptions

    # --- store the found value (lines 117-138) ---
    opt['Name'] = FoundName
    user_val    = AllOptions[FoundName]

    # struct merging (lines 118-135):
    # In MATLAB: triggered only when isstruct(value) is True.
    # uq_input objects are NOT plain structs in MATLAB — isstruct() returns False
    # for them, so they take the direct-assignment branch.
    # In Python both are dict; we guard merging on whether 'struct' is in
    # AllowedClasses, mimicking MATLAB's isstruct() distinction.
    _allowed_list = ([AllowedClasses] if isinstance(AllowedClasses, str)
                     else (AllowedClasses or []))
    _is_struct_context = 'struct' in _allowed_list

    if isinstance(user_val, dict) and _is_struct_context:
        # only update fields explicitly given; keep defaults for the rest
        set_options = list(user_val.keys())
        # if the default Value is not a dict, clear it (line 122-124)
        if not isinstance(opt['Value'], dict):
            opt['Value'] = {}
        if opt['Value'] is None:
            opt['Value'] = {}
        # copy each specified field into opt.Value (lines 127-130)
        for key in set_options:
            opt['Value'][key] = user_val[key]
    else:
        opt['Value'] = user_val

    opt['Type'] = FoundClass
    # remove the key from AllOptions (line 138)
    AllOptions  = {k: v for k, v in AllOptions.items() if k != FoundName}

    return opt, AllOptions


# ---------------------------------------------------------------------------
# uq_PCK_initialize
#   Faithful translation of PCK/uq_PCK_initialize.m
#
#   current_model : dict  (mirrors the MATLAB UQLab model object)
#     Required keys before call:
#       current_model['Internal']['Runtime']['M']  — number of input dimensions
#       current_model['Options']                    — user option dict
#     Optional key (must be present if Input not in Options):
#       current_model['Internal']['Runtime']['nonConstIdx']
#   global_input : dict | None
#     Equivalent of uq_getInput() — the session's current Input object.
#     Pass None if Input is always specified inside Options.
#
#   Returns True (success=1) or raises an exception on error (success=0).
# ---------------------------------------------------------------------------
def uq_PCK_initialize(current_model, global_input=None):
    """
    Initialise a PCK metamodel from current_model['Options'].
    Fills current_model['Internal'] with validated configuration.
    Word-for-word translation of uq_PCK_initialize.m.
    """
    # line 5 — success = 0  (implicit: exceptions replace success=0)

    # line 5  M = current_model.Internal.Runtime.M
    # (not used in this function beyond being available)
    # M = current_model['Internal']['Runtime']['M']   # kept for completeness

    # line 7  Options = current_model.Options
    Options = dict(current_model['Options'])   # shallow copy to avoid mutating

    # -----------------------------------------------------------------------
    # Default values  (lines 10-21)
    # -----------------------------------------------------------------------
    DEFAULTmode           = 'sequential'
    DEFAULTCombCrit       = 'rel_loo'
    DEFAULTIgnoreDependence = False

    DEFAULTKriging = {}    # MATLAB: DEFAULTKriging = []   (empty)

    DEFAULTPCE = {
        'MetaType': 'PCE',
        'Degree':   list(range(1, 4)),   # MATLAB: 1:3  →  [1, 2, 3]
        'Method':   'LARS',
    }

    # -----------------------------------------------------------------------
    # Input model  (lines 26-32)
    # -----------------------------------------------------------------------
    # if ~isfield(Options, 'Input')
    #     current_input = uq_getInput;
    # else
    #     current_input = [];
    # end
    if 'Input' not in Options:
        current_input = global_input   # uq_getInput equivalent
    else:
        current_input = None           # MATLAB: []

    input_opt, Options = uq_process_option(
        Options, 'Input', current_input, 'uq_input')
    current_model['Internal']['Input'] = input_opt['Value']

    # -----------------------------------------------------------------------
    # IgnoreDependence  (lines 36-58)
    # -----------------------------------------------------------------------
    IgnoreDependence, Options = uq_process_option(
        Options, 'IgnoreDependence', DEFAULTIgnoreDependence,
        ['logical', 'double'])

    if IgnoreDependence['Invalid']:
        raise ValueError('IgnoreDependence must be a logical!')
    else:
        # make sure an input is available (lines 41-43)
        inp = current_model['Internal'].get('Input')
        if inp is None or (hasattr(inp, '__len__') and len(inp) == 0):
            raise ValueError(
                'IgnoreDependence flag requested, but no Input object found')

        current_model['Internal']['IgnoreDependence'] = IgnoreDependence['Value']

        # substitute copula with independent one if requested (lines 49-57)
        if current_model['Internal']['IgnoreDependence']:
            copula_type = (current_model['Internal']['Input']
                           .get('Copula', {}).get('Type', 'independent'))
            if copula_type.lower() != 'independent':
                IndepInput = copy.deepcopy(current_model['Internal']['Input'])
                IndepInput['Copula']['Type']       = 'Independent'
                IndepInput['Copula']['Parameters'] = np.eye(
                    len(IndepInput['Marginals']))
                current_model['Internal']['Input'] = IndepInput

    # -----------------------------------------------------------------------
    # Mode: 'sequential' or 'optimal'  (lines 60-69)
    # -----------------------------------------------------------------------
    mode, Options = uq_process_option(Options, 'Mode', DEFAULTmode, 'char')

    # MATLAB: if strcmpi(mode.Value,'sequential') ... else if strcmpi(...,'optimal')
    if mode['Value'].lower() == 'sequential':
        current_model['Internal']['Mode'] = mode['Value']
    elif mode['Value'].lower() == 'optimal':
        current_model['Internal']['Mode'] = mode['Value']
    else:
        raise ValueError('something went wrong over here')

    # -----------------------------------------------------------------------
    # TrendMethod: 'user' or 'pce'  (lines 73-107)
    # -----------------------------------------------------------------------
    # if ~isfield(Options, 'PCE') && ~isfield(Options, 'PolyIndices')
    if 'PCE' not in Options and 'PolyIndices' not in Options:
        # nothing is specified -> use default PCE options
        current_model['Internal']['PCE']         = copy.copy(DEFAULTPCE)
        current_model['Internal']['TrendMethod'] = 'pce'

    else:
        # if ~isfield(Options, 'PCE') && isfield(Options, 'PolyIndices')
        if 'PCE' not in Options and 'PolyIndices' in Options:
            # only the polyindices and polytypes are given
            if 'PolyTypes' not in Options:
                raise ValueError('PolyTypes are missing')
            polyindices, Options = uq_process_option(
                Options, 'PolyIndices', None, 'double')
            current_model['Internal']['PolyIndices'] = polyindices['Value']
            polytypes, Options = uq_process_option(
                Options, 'PolyTypes', None, 'cell')
            current_model['Internal']['PolyTypes']   = polytypes['Value']
            current_model['Internal']['TrendMethod'] = 'user'

        else:
            # if isfield(Options, 'PCE') && ~isfield(Options, 'PolyIndices')
            if 'PCE' in Options and 'PolyIndices' not in Options:
                # PCE options but no polyindices
                pce, Options = uq_process_option(
                    Options, 'PCE', copy.copy(DEFAULTPCE), 'struct')
                current_model['Internal']['PCE']         = pce['Value']
                current_model['Internal']['TrendMethod'] = 'pce'
                # only LARS or OMP are available for PCK
                method = current_model['Internal']['PCE'].get('Method', '')
                if method.lower() not in ('lars', 'omp'):
                    raise ValueError(
                        'PCK only supports LARS or OMP sparse regression '
                        'methods for the PCE trend')
            else:
                # both PCE and PolyIndices are specified
                raise ValueError(
                    'options for pce and the set of polynomials cannot be '
                    'given at the same time')

    # -----------------------------------------------------------------------
    # CombCrit — only if mode='optimal'  (lines 111-114)
    # -----------------------------------------------------------------------
    if current_model['Internal']['Mode'].lower() == 'optimal':
        combo, Options = uq_process_option(
            Options, 'CombCrit', DEFAULTCombCrit, 'char')
        current_model['Internal']['CombCrit'] = combo['Value'].lower()

    # -----------------------------------------------------------------------
    # Kriging options  (lines 119-120)
    # -----------------------------------------------------------------------
    # [kriging, ~] = uq_process_option(Options, 'Kriging', DEFAULTKriging, 'struct')
    kriging, _ = uq_process_option(
        Options, 'Kriging', copy.copy(DEFAULTKriging), 'struct')
    current_model['Internal']['Kriging'] = kriging['Value']

    # -----------------------------------------------------------------------
    # Adjust Kriging.Optim.Bounds for constant inputs  (lines 124-129)
    # -----------------------------------------------------------------------
    # if ~isempty(current_model.Internal.Input.nonConst) &&
    #     isfield(current_model.Internal.Kriging,'Optim') &&
    #     isfield(current_model.Internal.Kriging.Optim,'Bounds')
    nonConst = (current_model['Internal'].get('Input') or {}).get('nonConst')
    krig_int = current_model['Internal'].get('Kriging', {})
    if (nonConst is not None
            and len(nonConst) > 0
            and 'Optim' in krig_int
            and 'Bounds' in krig_int['Optim']):
        # nonConst is a boolean or index array
        current_model['Internal']['Kriging']['Optim']['Bounds'] = \
            krig_int['Optim']['Bounds'][:, nonConst]

    # line 132  success = 1
    return True


# ---------------------------------------------------------------------------
# Helper: build a minimal current_model dict
# ---------------------------------------------------------------------------
def make_model(options, M, global_input=None):
    """
    Utility used in tests — creates the minimal current_model structure that
    uq_PCK_initialize expects (mirrors what uq_initialize_uq_metamodel builds).
    """
    model = {
        'Options': options,
        'Internal': {
            'Runtime': {'M': M},
            'Input':   global_input,
        },
    }
    return model
