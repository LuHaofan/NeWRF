function [step_length_vec, ix_pts_vec] = rayMarching(rays)
    step_length_vec = cell(length(rays),1);
    ix_pts_vec = cell(length(rays),1);
    for i = 1:length(rays)
        if rays(i).LineOfSight
            ix_pts = [rays(i).TransmitterLocation rays(i).ReceiverLocation];
        else
            ix_pts = [rays(i).TransmitterLocation rays(i).Interactions.Location rays(i).ReceiverLocation];
        end
        step_length_vec{i} = dist2PrevInteractionPt(ix_pts);
        ix_pts_vec{i} = ix_pts;
    end
end