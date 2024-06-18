function [lastXptlst, channelCoeffs] = helperGetRxChanPerRay(rays, fc, cfg, single_fc)
    n_paths = length(rays);
    [~, ix_pts] = rayMarching(rays);
    if single_fc
        lastXptlst = zeros(3, n_paths);
        channelCoeffs = zeros(1, n_paths);
        for i = 1:n_paths
            ray_pl = rays(i).PathLoss; 
            ray_pp = rays(i).PhaseShift;
            channelCoeffs(i) = 10^(-ray_pl/20).*exp(-1j*ray_pp);
            lastXptlst(:, i) = ix_pts{i}(:, end-1);
        end
        return
    end

    ofdmInfo = wlanNonHTOFDMInfo('L-LTF',cfg.ChannelBandwidth);
    sc_spacing = wlanSampleRate(cfg.ChannelBandwidth)/ofdmInfo.FFTLength;
    steps = fc+ofdmInfo.ActiveFrequencyIndices*sc_spacing;

    lastXptlst = zeros(3, n_paths);
    channelCoeffs = zeros(length(steps), n_paths);
    for i = 1:n_paths
        for j = 1:length(steps)
            fsc = steps(j);
            ray_pl = fspl(rays(i).PropagationDistance, fsc); %rays(i).PathLoss; 
            ray_pp = 2*pi*fsc*rays(i).PropagationDistance/physconst("lightspeed"); %rays(i).PhaseShift;
            channelCoeffs(j, i) = 10^(-ray_pl/20).*exp(-1j*ray_pp);
        end
        lastXptlst(:, i) = ix_pts{i}(:, end-1);
    end
end
