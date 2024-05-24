
function flag = isCollide(obj_dims, sta_loc)
    flag = 0;
    if sta_loc(1) > obj_dims(1,1) && ...
        sta_loc(1) < obj_dims(1,2) && ...
        sta_loc(2) > obj_dims(2,1) && ...
        sta_loc(2) < obj_dims(2,2) && ...
        sta_loc(3) > obj_dims(3,1) && ...
        sta_loc(3) < obj_dims(3,2)
        flag = 1;
    end
end