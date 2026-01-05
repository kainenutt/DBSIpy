default_dti_parameter_maps = {'dti_fa'                     : 'scalar',
                              'dti_cfa'                    : '3-vector',
                              'dti_axial'                  : 'scalar',
                              'dti_radial'                 : 'scalar',
                              'dti_adc'                    : 'scalar',
                              'b0_map'                     : 'scalar'}

expanded_dti_parameter_maps = {'dti_lambda_1'               : 'scalar',
                              'dti_lambda_2'               : 'scalar',
                              'dti_lambda_3'               : 'scalar',
                              'dti_eigenvec_1'             : '3-vector',
                              'dti_eigenvec_2'             : '3-vector',
                              'dti_eigenvec_3'             : '3-vector',
                              'dti_eigenvec_1_cfa'         : '3-vector',
                              'dti_eigenvec_2_cfa'         : '3-vector',
                              'dti_eigenvec_3_cfa'         : '3-vector'}


# Rich metadata (description/units/range/notes), matching NODDI metadata style.
from dbsipy.maps.map_metadata import get_map_spec


_ALL_DTI_MAPS = {**default_dti_parameter_maps, **expanded_dti_parameter_maps}

SCALAR_MAPS = {name: get_map_spec(name) for name, kind in _ALL_DTI_MAPS.items() if kind == 'scalar'}
VECTOR_MAPS = {name: get_map_spec(name) for name, kind in _ALL_DTI_MAPS.items() if kind != 'scalar'}


def get_map_info(map_name: str) -> dict:
    info = get_map_spec(map_name)
    if info is None:
        raise ValueError(f"Unknown DTI map: {map_name}")
    return info