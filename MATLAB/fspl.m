function pl = fspl(d, f)
    pl = 20*log10(d)+20*log10(f)+20*log10(4*pi/physconst("lightspeed"));
end