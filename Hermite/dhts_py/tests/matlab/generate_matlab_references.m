function generate_matlab_references(toolbox_dir, house_path, output_file)
%GENERATE_MATLAB_REFERENCES Export independent DHTS arrays for Python tests.
%
% generate_matlab_references(TOOLBOX_DIR, HOUSE_PATH, OUTPUT_FILE)
%
% TOOLBOX_DIR must contain the original unmodified .m files.  The output is
% MATLAB v7 format so scipy.io.loadmat can read it without extra packages.

if nargin < 1 || isempty(toolbox_dir)
    error('toolbox_dir is required and must point to the original DHTS toolbox');
end
if nargin < 2 || isempty(house_path)
    error('house_path is required and must point to house.tif');
end
if nargin < 3 || isempty(output_file)
    here = fileparts(mfilename('fullpath'));
    output_file = fullfile(here, 'references', 'dhts_matlab_reference.mat');
end

addpath(toolbox_dir);
cleanup = onCleanup(@() rmpath(toolbox_dir)); %#ok<NASGU>
output_dir = fileparts(output_file);
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% 1. DHTMTX: analysis and synthesis filters.
[H_N3_D3, G_N3_D3] = dhtmtx(3, 3, 1);
[H_N4_D4, G_N4_D4] = dhtmtx(4, 4, 1);
[H_N8_D3, G_N8_D3] = dhtmtx(8, 3, 1);
[H_N8_D8, G_N8_D8] = dhtmtx(8, 8, 1);

% 2. DHTORD.
ord_N8_D3 = dhtord(8, 3, 2);

% 3-5. Constant image and directional ramps.
constant_input = ones(40, 40);
constant_coefficients = dht2(constant_input, 8, 3, 1, 'symm');
ramp_horizontal_input = repmat(0:39, 40, 1);
ramp_horizontal_coefficients = dht2(ramp_horizontal_input, 8, 3, 1, 'symm');
ramp_vertical_input = repmat((0:39)', 1, 40);
ramp_vertical_coefficients = dht2(ramp_vertical_input, 8, 3, 1, 'symm');

% 6-9. Real image, gauge, RDHT, and inverse RDHT.
house_raw = imread(house_path);
house_input = double(house_raw) / 255;
if ndims(house_input) == 3 && size(house_input, 3) >= 3
    house_input = 0.298936021293775 * house_input(:,:,1) ...
                + 0.587043074451121 * house_input(:,:,2) ...
                + 0.114020904255103 * house_input(:,:,3);
elseif ndims(house_input) == 3
    house_input = house_input(:,:,1);
end
house_coefficients = dht2(house_input, 8, 3, 1, 'symm');
theta_gauge = gauge(house_coefficients, 8, 3, 1);
rdht_forward = rdht(house_coefficients, 8, 3, 'fwd', theta_gauge);
rdht_inverse = rdht(rdht_forward, 8, 3, 'inv', theta_gauge);

% 10. Complete (D=16) and explicitly truncated (D=3) reconstruction.
[xx, yy] = meshgrid(linspace(-1, 1, 32), linspace(-1, 1, 32));
reconstruction_input = sin(2.3 * xx) + 0.4 * cos(1.7 * yy) + 0.2 * xx .* yy;
reconstruction_complete_coefficients = dht2(reconstruction_input, 8, 16, 1, 'full');
reconstruction_complete = idht2(reconstruction_complete_coefficients, size(reconstruction_input), 8, 16, 1, 'full');
reconstruction_truncated_coefficients = dht2(reconstruction_input, 8, 3, 1, 'full');
reconstruction_truncated = idht2(reconstruction_truncated_coefficients, size(reconstruction_input), 8, 3, 1, 'full');

save(output_file, ...
    'H_N3_D3', 'G_N3_D3', 'H_N4_D4', 'G_N4_D4', ...
    'H_N8_D3', 'G_N8_D3', 'H_N8_D8', 'G_N8_D8', ...
    'ord_N8_D3', ...
    'constant_input', 'constant_coefficients', ...
    'ramp_horizontal_input', 'ramp_horizontal_coefficients', ...
    'ramp_vertical_input', 'ramp_vertical_coefficients', ...
    'house_input', 'house_coefficients', 'theta_gauge', ...
    'rdht_forward', 'rdht_inverse', ...
    'reconstruction_input', ...
    'reconstruction_complete_coefficients', 'reconstruction_complete', ...
    'reconstruction_truncated_coefficients', 'reconstruction_truncated', ...
    '-v7');

fprintf('MATLAB references written to %s\n', output_file);
end
