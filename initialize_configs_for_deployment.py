## Python script to initialize configuration templates for release version of DBSIpy

import os
import glob
import natsort
import configparser

def main():
    script_path     = os.path.abspath(__file__)
    cfg_dir_path    = os.path.split(script_path)[0] + os.sep + 'dbsipy' + os.sep + 'configs'
    cfg_templates   = natsort.natsorted(glob.glob(cfg_dir_path + os.sep + '*.ini'))

    for iCfg in cfg_templates:
        config = configparser.ConfigParser()
        config.read(iCfg)

        angle_basis_path    = config.get('STEP_1','angle_basis')
        iso_basis_path      = config.get('STEP_1','iso_basis')
        axial_basis_path    = config.get('STEP_2','step_2_axials')
        radial_basis_path   = config.get('STEP_2','step_2_radials')

        angle_basis_path    = cfg_dir_path + os.sep + 'BasisSets' + os.sep + angle_basis_path.split('BasisSets/',1)[1].split('/',1)[0] + os.sep + os.path.split(angle_basis_path)[1]
        iso_basis_path      = cfg_dir_path + os.sep + 'BasisSets' + os.sep + iso_basis_path.split('BasisSets/',1)[1].split('/',1)[0] + os.sep + os.path.split(iso_basis_path)[1]
        axial_basis_path    = cfg_dir_path + os.sep + 'BasisSets' + os.sep + axial_basis_path.split('BasisSets/',1)[1].split('/',1)[0] + os.sep + os.path.split(axial_basis_path)[1]
        radial_basis_path   = cfg_dir_path + os.sep + 'BasisSets' + os.sep + radial_basis_path.split('BasisSets/',1)[1].split('/',1)[0] + os.sep + os.path.split(radial_basis_path)[1]

        config.set('STEP_1','angle_basis',str(angle_basis_path))
        config.set('STEP_1','iso_basis',str(iso_basis_path))
        config.set('STEP_2','step_2_axials',str(axial_basis_path))
        config.set('STEP_2','step_2_radials',str(radial_basis_path))
        
        with open(iCfg, 'w') as configfile:
            config.write(configfile)




if __name__ == "__main__":
    main()