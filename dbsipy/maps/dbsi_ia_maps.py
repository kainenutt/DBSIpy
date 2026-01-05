default_dbsi_ia_parameter_maps = {  'isotropic_adc'             : 'scalar',
                                    'isotropic_fraction'        : 'scalar',
                                    'b0_map'                    : 'scalar',
                                    'fiber_0d_fraction'         : 'scalar',
                                    'fiber_0d_fa'               : 'scalar',
                                    'fiber_0d_axial'            : 'scalar',
                                    'fiber_0d_radial'           : 'scalar',
                                    'fiber_0d_adc'              : 'scalar',
                                    'fiber_0d_cfa'              : '3-vector',
                                    'fiber_0d_IA_fraction'      : 'scalar',
                                    'fiber_0d_IA_fa'            : 'scalar',
                                    'fiber_0d_IA_axial'         : 'scalar',
                                    'fiber_0d_IA_radial'        : 'scalar',
                                    'fiber_0d_IA_adc'           : 'scalar',
                                    'fiber_0d_EA_fraction'      : 'scalar',
                                    'fiber_0d_EA_fa'            : 'scalar',
                                    'fiber_0d_EA_axial'         : 'scalar',
                                    'fiber_0d_EA_radial'        : 'scalar',
                                    'fiber_0d_EA_adc'           : 'scalar'}

expanded_dbsi_ia_parameter_maps     = { 'fiber_0d_lambda_1'          : 'scalar',
                                        'fiber_0d_lambda_2'          : 'scalar',
                                        'fiber_0d_lambda_3'          : 'scalar',
                                        'fiber_0d_eigenvec_1'        : '3-vector',
                                        'fiber_0d_eigenvec_2'        : '3-vector',
                                        'fiber_0d_eigenvec_3'        : '3-vector',
                                        'fiber_0d_eigenvec_1_cfa'    : '3-vector',
                                        'fiber_0d_eigenvec_2_cfa'    : '3-vector',
                                        'fiber_0d_eigenvec_3_cfa'    : '3-vector',
                                        'fiber_0d_IA_lambda_1'       : 'scalar',
                                        'fiber_0d_IA_lambda_2'       : 'scalar',
                                        'fiber_0d_IA_lambda_3'       : 'scalar',
                                        'fiber_0d_IA_eigenvec_1'     : '3-vector',
                                        'fiber_0d_IA_eigenvec_2'     : '3-vector',
                                        'fiber_0d_IA_eigenvec_3'     : '3-vector',
                                        'fiber_0d_IA_eigenvec_1_cfa' : '3-vector',
                                        'fiber_0d_IA_eigenvec_2_cfa' : '3-vector',
                                        'fiber_0d_IA_eigenvec_3_cfa' : '3-vector',
                                        'fiber_0d_EA_lambda_1'       : 'scalar',
                                        'fiber_0d_EA_lambda_2'       : 'scalar',
                                        'fiber_0d_EA_lambda_3'       : 'scalar',
                                        'fiber_0d_EA_eigenvec_1'     : '3-vector',
                                        'fiber_0d_EA_eigenvec_2'     : '3-vector',
                                        'fiber_0d_EA_eigenvec_3'     : '3-vector',
                                        'fiber_0d_EA_eigenvec_1_cfa' : '3-vector',
                                        'fiber_0d_EA_eigenvec_2_cfa' : '3-vector',
                                        'fiber_0d_EA_eigenvec_3_cfa' : '3-vector'}

three_seg_iso_maps     = {'restricted_adc'             : 'scalar', 
                          'restricted_fraction'        : 'scalar',
                          'hindered_fraction'          : 'scalar',
                          'hindered_adc'               : 'scalar',
                          'water_fraction'             : 'scalar',
                          'water_adc'                  : 'scalar'}

four_seg_iso_maps     = { 'highly_restricted_adc'      : 'scalar', 
                          'highly_restricted_fraction' : 'scalar',
                          'restricted_adc'             : 'scalar', 
                          'restricted_fraction'        : 'scalar',
                          'hindered_fraction'          : 'scalar',
                          'hindered_adc'               : 'scalar',
                          'water_fraction'             : 'scalar',
                          'water_adc'                  : 'scalar'}


# Rich metadata (description/units/range/notes), matching NODDI metadata style.
from dbsipy.maps.map_metadata import get_map_spec


_ALL_IA_MAPS = {
    **default_dbsi_ia_parameter_maps,
    **expanded_dbsi_ia_parameter_maps,
    **three_seg_iso_maps,
    **four_seg_iso_maps,
}

SCALAR_MAPS = {name: get_map_spec(name) for name, kind in _ALL_IA_MAPS.items() if kind == 'scalar'}
VECTOR_MAPS = {name: get_map_spec(name) for name, kind in _ALL_IA_MAPS.items() if kind != 'scalar'}


def get_map_info(map_name: str) -> dict:
    info = get_map_spec(map_name)
    if info is None:
        raise ValueError(f"Unknown DBSI-IA map: {map_name}")
    return info

