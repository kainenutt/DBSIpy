function DBSIpy_fib(results_dir)
%-AUTHOR-----------------------------------------------------------------------------
% Kainen L. Utt, PhD
% Postdoctoral Research Associate
% Biomedical MR Center, Malinckrodt Institute of Radiology
% Washington University School of Medicine in St. Louis, St. Louis, MO 63110, USA
% k.l.utt at wustl dot edu
%------------------------------------------------------------------------------------

if nargin < 1
    resDir   = uigetdir(pwd,'Select Results Directory');
    if isequal(resDir,0) || isempty(resDir)
        disp('User pressed cancel')
        return
    end

    prompt = {'x:','y','z'};
    dlg_title = 'image resolution to fib';
    num_lines = 1;
    def = {'1','1','1'};
    answer = inputdlg(prompt,dlg_title,num_lines,def);
    output_fib_res = [str2double(answer{1}) str2double(answer{2}) str2double(answer{3})]; 
else
    resDir = char(results_dir);
    output_fib_res = [1 1 1];
end

b0_map              = double(niftiread([resDir filesep 'b0_map.nii.gz']));
dti_axial_map       = double(niftiread([resDir filesep 'dti_ad_map.nii.gz']));
dti_adc_map         = double(niftiread([resDir filesep 'dti_adc_map.nii.gz']));
dti_orientation_map = double(niftiread([resDir filesep 'dti_direction_map.nii.gz']));
dti_fa_map          = double(niftiread([resDir filesep 'dti_fa_map.nii.gz']));
dti_radial_map      = double(niftiread([resDir filesep 'dti_rd_map.nii.gz']));
f1_axial_map        = double(niftiread([resDir filesep 'fiber_01_ad_map.nii.gz']));
f1_adc_map          = double(niftiread([resDir filesep 'fiber_01_adc_map.nii.gz']));
f1_orientation_map  = double(niftiread([resDir filesep 'fiber_01_direction_map.nii.gz']));
f1_fa_map           = double(niftiread([resDir filesep 'fiber_01_fa_map.nii.gz']));
f1_fraction_map     = double(niftiread([resDir filesep 'fiber_01_fraction_map.nii.gz']));
f1_radial_map       = double(niftiread([resDir filesep 'fiber_01_rd_map.nii.gz']));
res_frac_map        = double(niftiread([resDir filesep 'restricted_fraction_map.nii.gz']));
hindered_frac_map   = double(niftiread([resDir filesep 'hindered_fraction_map.nii.gz']));
water_frac_map      = double(niftiread([resDir filesep 'water_fraction_map.nii.gz']));
res_adc_map         = double(niftiread([resDir filesep 'restricted_adc_map.nii.gz']));
hindered_adc_map    = double(niftiread([resDir filesep 'hindered_adc_map.nii.gz']));
water_adc_map       = double(niftiread([resDir filesep 'water_adc_map.nii.gz']));


%% DTI Fiber Tracking Results

fib.dimension = size(b0_map);
fib.voxel_size = output_fib_res;

fib.dir0 = zeros([3 fib.dimension]);
fib.dir0 = permute(dti_orientation_map,[4,1,2,3]); fib.dir0 = reshape(fib.dir0,1,[]);
fib.b0 = b0_map; fib.b0 = reshape(fib.b0,1,[]);  
fib.fa0 = dti_fa_map; fib.fa0 = reshape(fib.fa0,1,[]); fib.fa0(isnan(fib.fa0))=0;
fib.dti_fa = dti_fa_map; fib.dti_fa = reshape(fib.dti_fa,1,[]);  fib.dti_fa(isnan(fib.dti_fa))=0;
fib.dti_axial = dti_axial_map; fib.dti_axial = reshape(fib.dti_axial,1,[]);
fib.dti_radial = dti_radial_map; fib.dti_radial = reshape(fib.dti_radial,1,[]);    
fib.dti_adc = dti_adc_map; fib.dti_adc = reshape(fib.dti_adc,1,[]);  

% save into FIB file
save(fullfile(resDir,'dti_tracking.fib'),'-struct','fib','-v4');      

%% DBSIpy Fiber Tracking Results
fib.dti_fa = dti_fa_map;
fib.dti_fa = reshape(fib.dti_fa,1,[]);
fib.dti_fa(isnan(fib.dti_fa))=0;
fib.dti_axial = dti_axial_map;
fib.dti_axial = reshape(fib.dti_axial,1,[]);
fib.dti_radial = dti_radial_map;
fib.dti_radial = reshape(fib.dti_radial,1,[]);   
fib.dti_adc = dti_adc_map;
fib.dti_adc = reshape(fib.dti_adc,1,[]);    

fib.dir0 = zeros([3 fib.dimension]);
fib.dir0 = permute(f1_orientation_map,[4,1,2,3]);
fib.dir0 = reshape(fib.dir0,1,[]);

fib.fa0 = f1_fa_map;
fib.fa0 = reshape(fib.fa0,1,[]);
fib.fr0 = f1_fraction_map;
fib.fr0 = reshape(fib.fr0,1,[]);
fib.fiber1_axial = f1_axial_map;
fib.fiber1_axial = reshape(fib.fiber1_axial,1,[]);
fib.fiber1_radial = f1_radial_map;
fib.fiber1_radial = reshape(fib.fiber1_radial,1,[]);
fib.fiber1_adc = f1_adc_map;
fib.fiber1_adc = reshape(fib.fiber1_adc,1,[]);
fib.restricted_fraction = res_frac_map;
fib.restricted_fraction = reshape(fib.restricted_fraction,1,[]);
fib.hindered_fraction = hindered_frac_map;
fib.hindered_fraction = reshape(fib.hindered_fraction,1,[]);
fib.water_fraction = water_frac_map;
fib.water_fraction = reshape(fib.water_fraction,1,[]);  

fib.restricted_adc = res_adc_map;
fib.restricted_adc = reshape(fib.restricted_adc,1,[]);
fib.hindered_adc = hindered_adc_map;
fib.hindered_adc = reshape(fib.hindered_adc,1,[]);
fib.water_adc = water_adc_map;
fib.water_adc = reshape(fib.water_adc,1,[]);  

fib.fiber_fraction = fib.fr0;

save(fullfile(resDir,'DBSIpy_tracking.fib'),'-struct','fib','-v4'); 

end



