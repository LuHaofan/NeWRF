function ret = dist2PrevInteractionPt(ix_pts)
% ix_pts: the array of interaction points locations, in shape: [3, N_pt]
    ret = zeros(1, size(ix_pts,2)-1);
    for i = 2:size(ix_pts,2)
        ret(i-1) = distance3D(ix_pts(:,i), ix_pts(:,i-1));
    end
end