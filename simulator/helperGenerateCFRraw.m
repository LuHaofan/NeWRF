function cfr = helperGenerateCFRraw(rays, fc, cfg, single_fc)
%GENERATECFR Generate Channel Frequency Response forthe simulated
%ray-tracing channel.

    n_paths = length(rays);
    coeffs = zeros(1, n_paths);
    if single_fc
        for j = 1:n_paths
            ray_pl = rays(j).PathLoss;
            ray_pp = rays(j).PhaseShift; 
            coeffs(j) = 10^(-ray_pl/20).*exp(-1j*ray_pp);
        end
        cfr = sum(coeffs);
        return
    end


    ofdmInfo = wlanNonHTOFDMInfo('L-LTF',cfg.ChannelBandwidth);
    sc_spacing = wlanSampleRate(cfg.ChannelBandwidth)/ofdmInfo.FFTLength;
    subcarrier_freq = fc+ofdmInfo.ActiveFrequencyIndices*sc_spacing;

    cfr = zeros(length(subcarrier_freq),1);
    for i = 1:length(subcarrier_freq)
        fsc = subcarrier_freq(i);
        coeffs = zeros(1, n_paths);
        for j = 1:n_paths
            ray_pl =  fspl(rays(j).PropagationDistance, fsc);%rays(j).PathLoss;
            ray_pp = 2*pi*fsc*rays(j).PropagationDistance/physconst("lightspeed"); %rays(j).PhaseShift; 
            coeffs(j) = 10^(-ray_pl/20).*exp(-1j*ray_pp);
        end
        cfr(i) = sum(coeffs);
    end
end