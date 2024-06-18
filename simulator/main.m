close all; clear; clc;

mapFileName = "models/conferenceroom.stl";
viewer = siteviewer("SceneModel",mapFileName,"Transparency",0.25);
%% Measure Map Dimensions
[vertex,face] = stlread(mapFileName);
xy_offset = 0.1;
z_offset = 0.1;
x_range = [min(vertex.Points(:,1))+xy_offset, max(vertex.Points(:,1))-xy_offset];
y_range = [min(vertex.Points(:,2))+xy_offset, max(vertex.Points(:,2))-xy_offset];
z_range = [min(vertex.Points(:,3)), max(vertex.Points(:,3))-z_offset];
env_dims = [x_range; y_range; z_range];

%% System configuration
cfg = wlanNonHTConfig();

%% Setup AP
fc = 2.412e9;
antPosAP = [-1.5; 0.0; 1.7];    % for conference room env
% antPosAP = [-1.5; 0.0; 2.7];  % for bedroom env
AP = txsite("cartesian", ...
    "AntennaPosition", antPosAP,...
    "TransmitterFrequency", fc, ...
    "TransmitterPower",1);

%% Setup STAs
distribution = "random";  % "uniform" | "random"
staSeparation = 0.5; % STA separation, in meters, used only when the distribution is uniform
numSTAs = 500;
S = RandStream("mt19937ar","Seed",5489); % Set the RNG for reproducibility.
RandStream.setGlobalStream(S);
if distribution == "uniform"
    STAs = createSTAs(env_dims,staSeparation,"uniform");
else
    STAs = createSTAs(env_dims,numSTAs,"random");
end
numAPs = 1;
numSTAs = length(STAs);


%% Visualize the environment
% show(AP)
% show(STAs,"ShowAntennaHeight",false,"IconSize",[16 16]);

%% Run ray tracing simulation
pm = propagationModel("raytracing", ...
            "Method","image", ...
            "CoordinateSystem","cartesian", ...
            "SurfaceMaterial","perfect-reflector", ...
            "MaxNumReflections",2); 
rays = raytrace(AP,STAs,pm,"Map",mapFileName);
numChan = numel(rays);

%% Store the simulation data
dataset = [];
blocklos = false;
single_fc = true;       % if false, generate for all subcarriers
disp(['Simulating Frequency: ', num2str(fc/1e9), ' GHz'])
subdataset = struct('TxPos', cell(numChan,1), 'TxID', cell(numChan,1), ...
    'RxPos',cell(numChan,1), 'RxID', cell(numChan,1), ...
    'Frequency', cell(numChan,1), 'CSI', cell(numChan,1), ...
    'LastXPts', cell(numChan, 1), ...
    'RxChanPerRay', cell(numChan, 1), ...
    'LineOfSight', cell(numChan, 1));

for i = 1:numChan
    if ~isempty(rays{i})
        % Generates the channel estimate/returns the CFR/CIR.
        cfrRaw = helperGenerateCFRraw(rays{i},fc,cfg, single_fc);
        [lastXptLst, rxChanPerRay] = helperGetRxChanPerRay(rays{i}, fc, cfg, single_fc);
        txpos = [AP.AntennaPosition];
        rxpos = [STAs(i).AntennaPosition];
        subdataset(i).TxPos = txpos;
        subdataset(i).TxID = 1;
        subdataset(i).RxPos = rxpos;
        subdataset(i).RxID = i;
        subdataset(i).Frequency = fc/1e9;
        subdataset(i).CSI = cfrRaw;
        subdataset(i).LastXPts = lastXptLst;
        subdataset(i).RxChanPerRay = rxChanPerRay;
        subdataset(i).LineOfSight = rays{i}(1).LineOfSight;
    end       

    % Displays progress (10% intervals)
    if mod(i,floor(numChan/10))==0
        qt = ceil(i/(numChan/10));
        disp(['Generating Dataset: ', num2str(10*qt), '% complete.'])
    end
end
save("datasets/dataset_conference.mat", 'subdataset');

