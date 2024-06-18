function d = distance3D(loc1, loc2)
    d = sqrt(sum((loc1-loc2).^2));
end