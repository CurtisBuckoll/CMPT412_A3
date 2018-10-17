close all;

% ----------------------
% Weird things to remember to look at later:
%
% You have the loop currently constructing the lookup table from pixels
% close starting at y = 480 down to y = 1
%
% ----------------------

ref1 = rgb2gray(imread('./synth/sphere1.tif'));
ref2 = rgb2gray(imread('./synth/sphere2.tif'));
ref3 = rgb2gray(imread('./synth/sphere3.tif'));

[H, W, C] = size(ref1);
ref_imgs = {ref1, ref2, ref3};

centerX = 320;
centerY = 240;
centerZ = 0;
r       = 152;

% imshow(ref1);
% imshow(ref2);
% imshow(ref3);
% imshow(ref1);

ratio_data1 = zeros(2,1);
ratio_data2 = zeros(2,1);
i = 1;

LUT_SZ = 10000;

% ----------------------

s_i = get_s_vectors(centerX, centerY, centerZ, r, ref_imgs);
% [H, W, C] = size(ref1);
% [row,col] = find( ref1 >= 255 );
% avg_y = round(sum(row) / size(row,1));
% avg_x = round(sum(col) / size(col,1));
% lin_ind = col + (row) * H;
% %ref1(find( ref1 >= 254 )) = 0;
% imshow(ref1);
% ref1(avg_y,avg_x) = 0;
% 
% %find( ref1 == 255 )
% %ref1(find( ref1 == 255 )) = 0;
% 
% imshow(ref1);

% ----------------------

PERCENTILES = generate_indices(H, W, s_i, centerX, centerY, centerZ, r, LUT_SZ);

% We need to reset the lookup table size, since generate_indices()
% may return only approximately the requested indices.
LUT_SZ = size(PERCENTILES, 2);
plot(1:LUT_SZ, PERCENTILES);

LUT = zeros(LUT_SZ, LUT_SZ, 2);

% ----------------------

% We must assume all images have the same size.

im = zeros(H,W,3);
for YY=1:H
    y = (H+1)-YY;
    for x=1:W
        if ( sqrt((x-centerX)^2 + (y-centerY)^2) <= r )
            
%             % Get the surface normal n as unit vec and ad we understand 
%             % ithe orientation w.r.t the sphere. 'z' should be -ve here. (?)
%             z = sqrt( r^2 - (x - centerX)^2 - (y - centerY)^2 );
%             n = [x; y; -z] - [centerX; centerY; centerZ];
%             n = n ./ norm(n);
% 
%             % If the z component is 0, add a small delta and renormalize.
%             DELTA = 0.01;
%             if ( n(3) == 0 )
%                n = n + DELTA;
%                n = n / norm(n);
%                %continue;
%             end
% 
%             % Obtain the 'gradient' vector here: (p,q,-1).
%             grad = n ./ -n(3);

            n = get_surface_normal(x, y, r, centerX, centerY, centerZ);
            
            if (ref2(y,x) ~= 0 || ref3(y,x) ~= 0)
                % Get the pixel ratios across the three calibration images.
                r1 = (double(ref1(y,x)) + 1) / (double(ref2(y,x)) + 1);
                r2 = (double(ref2(y,x)) + 1) / (double(ref3(y,x)) + 1);

                % Get the indices of the nearest ratios mapped to integer
                % indices which we will use to store in the lookup table.
                [approx_r1, i_1] = min(abs(PERCENTILES - r1));
                [approx_r2, i_2] = min(abs(PERCENTILES - r2));
                PERCENTILES(i_1);
                PERCENTILES(i_2);

                % Store in the table.
                if ( LUT(i_1, i_2, 1) == 0 && LUT(i_1, i_2, 2) == 0 ) 
                    LUT(i_1, i_2, 1) = n(1);
                    LUT(i_1, i_2, 2) = n(2);
                end
%                 else % - I don't think we want to do this at all.
%                     % average
%                     LUT(i_1, i_2, 1) = (grad(1) + LUT(i_1, i_2, 1)) / 2;
%                     LUT(i_1, i_2, 2) = (grad(2) + LUT(i_1, i_2, 2)) / 2;
%                 end

                im(y,x,:) = n;
            end
        end
    end
end

display_gradient(im, 20);

% im = zeros(H,W,3);
% for y=1:H
%     for x=1:W
%         if ( sqrt((x-centerX)^2 + (y-centerY)^2) <= r )
% 
%             % Get the intensity ratios then look up the corresponding
%             % gradient vector in the LUT.
%             r1 = (double(ref1(y,x)) + 1) / (double(ref2(y,x)) + 1);
%             r2 = (double(ref2(y,x)) + 1) / (double(ref3(y,x)) + 1);
%             
%             % Get the indices of the pixel intensity ratios.
%             [approx_r1, i_1] = min(abs(PERCENTILES - r1));
%             [approx_r2, i_2] = min(abs(PERCENTILES - r2));
%             PERCENTILES(i_1);
%             PERCENTILES(i_2);
%            
%             grad = [LUT(i_1, i_2, 1), LUT(i_1, i_2, 2), -1];
%             
%             im(y,x,:) = grad;
%             
%         end
%     end
% end

display_gradient(im, 20);



% ------------------------------------------------------------------
% 
function res = display_gradient(grad, sz)
    [h, w, c] = size(grad);
    avg_horiz = grad((1:sz:h),:,:);
    for i=1:sz-1
        avg_horiz = avg_horiz + grad((1:sz:h)+i,:,:);
    end
    res = avg_horiz(:,(1:sz:w),:);
    for i=1:sz
        res = res + avg_horiz(:,(1:sz:w)+i-1,:);
    end
    res = res ./ sz^2;
    figure
    quiver(res(:,:,1), res(:,:,2));
end

% ------------------------------------------------------------------
% 
function percentiles = generate_indices(imH, imW, s_i, centerX, centerY, centerZ, r, num_entries)

    % We must assume all images have the same size.
    ratio_data = zeros(1,1);
    i = 1;
    s_1 = s_i(1,:);
    s_2 = s_i(2,:);
    s_3 = s_i(3,:);

    for y=1:imH
        for x=1:imW
            if ( sqrt((x-centerX)^2 + (y-centerY)^2) <= r )

%                 % Get the surface normal n as unit vec and as we understand 
%                 % the orientation w.r.t the sphere. 'z' should be -ve here. (?)
%                 % Note - This work will be reperformed when we actually input 
%                 % the data into table, but we also need it here to determine
%                 % the distribution of the table entries.
%                 z = sqrt( r^2 - (x - centerX)^2 - (y - centerY)^2 );
%                 n = [x; y; -z] - [centerX; centerY; centerZ];
%                 n = n ./ norm(n);
% 
%                 % If the z component is 0, add a small delta and renormalize.
%                 DELTA = 0.01;
%                 if ( n(3) == 0 )
%                    n = n + DELTA;
%                    n = n / norm(n);
%                    %continue;
%                 end
% 
%                 % Obtain the 'gradient' vector here: (p,q,-1).
%                 grad = n ./ -n(3);
                n = get_surface_normal(x, y, r, centerX, centerY, centerZ);
                
                % Now we can compute the 'intensities' from the dot
                % products of the light intensities (s_i) and the supposed
                % surface normal (n).
                E_1 = dot(n, s_i(1,:));
                E_2 = dot(n, s_i(2,:));
                E_3 = dot(n, s_i(3,:));
                
                if ( E_2 > 0 || E_3 > 0 )
                    r1 = ((E_1 * 255) + 1) / ((E_2 * 255) + 1);
                    r2 = ((E_2 * 255) + 1) / ((E_3 * 255) + 1);

                    ratio_data(i) = r1;
                    ratio_data(i+1) = r2;

                    i = i + 2;
                end
            end
        end
    end
    
    p =(1:100/num_entries:100);
    percentiles = prctile(ratio_data,p);
end

% ------------------------------------------------------------------
% 
function percentiles = generate_indices_old(im1, im2, im3, centerX, centerY, r, num_entries)

    % We must assume all images have the same size.
    [H, W, C]  = size(im1);
    ratio_data = zeros(1,1);
    i = 1;

    for y=1:H
        for x=1:W
            if ( sqrt((x-centerX)^2 + (y-centerY)^2) <= r )

                r1 = (double(im1(y,x)) + 1) / (double(im2(y,x)) + 1);
                r2 = (double(im2(y,x)) + 1) / (double(im3(y,x)) + 1);

                ratio_data(i) = r1;
                ratio_data(i+1) = r2;

                i = i + 2;
            end
        end
    end
    
    p =(1:100/num_entries:100);
    percentiles = prctile(ratio_data,p);
end

% ------------------------------------------------------------------
% Takes grayscale images, assumes all images in ref_imgs are the
% same size. Returns a 3x3 matrix where each row corresponds to
% s_1, s_2, s_3 resp.
function s_i = get_s_vectors(centerX, centerY, centerZ, r, ref_imgs)

    s_i = zeros(3,3);
    
    for i=1:3
        img = ref_imgs{i};
        
        % Find the brightest point on the sphere and average the
        % coordinates.
        [row,col] = find( img >= 255 );
        avg_y = round(sum(row) / size(row,1));
        avg_x = round(sum(col) / size(col,1));
        
%         imshow(img);
%         img(avg_y,avg_x) = 0;
%         imshow(img);
        
        z = sqrt( r^2 - (avg_x - centerX)^2 - (avg_y - centerY)^2 );
        s = [avg_x; avg_y; -z] - [centerX; centerY; centerZ];
        s_i(i,:) = s ./ norm(s);
    end
end

% ------------------------------------------------------------------
%
function grad = get_surface_normal(x, y, r, centerX, centerY, centerZ)
    %
    % Get the surface normal/gradient n as unit vec and as we understand 
    % the orientation w.r.t the sphere. 'z' should be -ve here. (?)
    %
    % Returns the surface normal (p,q,-1) where (p,q) is the gradient
    % at the supplied point.
    %
    % Note - This work will be reperformed when we actually input 
    % the data into table, but we also need it here to determine
    % the distribution of the table entries.
    
    DELTA = 0.01;
    z = sqrt( r^2 - (x - centerX)^2 - (y - centerY)^2 );
    n = [x; y; -z] - [centerX; centerY; centerZ];
    n = n ./ norm(n);

    % If the z component is 0, add a small delta and renormalize.
    if ( n(3) == 0 )
       n = n + DELTA;
       n = n / norm(n);
       %continue;
    end

    % Obtain the 'gradient' vector here: (p,q,-1).
    grad = n ./ -n(3);
end




