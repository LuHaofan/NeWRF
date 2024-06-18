function STAs = createSTAs(env_dims, val, distribution)
    % Define valid space for STAs
    xSTA = env_dims(1,:);
    ySTA = env_dims(2,:);
    zSTA = env_dims(3,:);
    
    dX = diff(xSTA);
    dY = diff(ySTA);
    dZ = diff(zSTA);
    dims = [dX dY dZ];

    if distribution=="uniform"
        % Create uniform grid within bounded range of valid STA locations
        
        % Offset each dimension so grid is centered
        rxSep = val;
        numSeg = floor(dims/rxSep);
        dimsOffset = (dims-(numSeg*rxSep))./2;  
        xGridSTA = (min(xSTA)+dimsOffset(1)):rxSep:(max(xSTA)-dimsOffset(1));
        yGridSTA = (min(ySTA)+dimsOffset(2)):rxSep:(max(ySTA)-dimsOffset(2));
        zGridSTA = (min(zSTA)+dimsOffset(3)):rxSep:(max(zSTA)-dimsOffset(3));
        
        % Set the position of the STA antenna centroid by replicating the
        % Position vectors across 3D space.
        antPosSTA = [repmat(kron(xGridSTA, ones(1, length(yGridSTA))), 1, length(zGridSTA)); ...
                  repmat(yGridSTA, 1, length(xGridSTA)*length(zGridSTA)); ...
                  kron(zGridSTA, ones(1, length(yGridSTA)*length(xGridSTA)))];
    else 
        % Randomly assign n STA positions bounded by the range of valid STA locations
        numSTA = val;
        antPosSTA = [((max(xSTA)-min(xSTA)).*rand(numSTA, 1)+min(xSTA))';
                    ((max(ySTA)-min(ySTA)).*rand(numSTA, 1)+min(ySTA))';
                    ((max(zSTA)-min(zSTA)).*rand(numSTA, 1)+min(zSTA))'];
    end

    STAs = rxsite("cartesian", ...
    "AntennaPosition", antPosSTA, ...
    "AntennaAngle", 0, ...
    "ReceiverSensitivity",-85);
end