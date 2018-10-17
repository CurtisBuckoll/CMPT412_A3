close all;

ref1 = rgb2gray(imread('./synth/sphere1.tif'));
ref2 = rgb2gray(imread('./synth/sphere2.tif'));
ref3 = rgb2gray(imread('./synth/sphere3.tif'));

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

get_s_vectors(centerX, centerY, centerZ, r, ref_imgs);
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

PERCENTILES = generate_indices(ref1, ref2, ref3, centerX, centerY, r, LUT_SZ);

% We need to reset the lookup table size, since generate_indices()
% may return only approximately the requested indices.
LUT_SZ = size(PERCENTILES, 2);
plot(1:LUT_SZ, PERCENTILES);

LUT = zeros(LUT_SZ, LUT_SZ, 2);

% ----------------------

% We must assume all images have the same size.
[H, W, C] = size(ref1);
im = zeros(H,W,3);
for YY=1:H
    y = (H+1)-YY;
    for x=1:W
        if ( sqrt((x-centerX)^2 + (y-centerY)^2) <= r )
            
            % Get the surface normal n as unit vec and ad we understand 
            % ithe orientation w.r.t the sphere. 'z' should be -ve here. (?)
            z = sqrt( r^2 - (x - centerX)^2 - (y - centerY)^2 );
            n = [x; y; -z] - [centerX; centerY; centerZ];
            n = n ./ norm(n);

            % If the z component is 0, add a small delta and renormalize.
            DELTA = 0.01;
            if ( n(3) == 0 )
               n = n + DELTA;
               n = n / norm(n);
               %continue;
            end

            % Obtain the 'gradient' vector here: (p,q,-1).
            grad = n ./ -n(3);
            
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
                    LUT(i_1, i_2, 1) = grad(1);
                    LUT(i_1, i_2, 2) = grad(2);
                else
%                     % average
%                     LUT(i_1, i_2, 1) = (grad(1) + LUT(i_1, i_2, 1)) / 2;
%                     LUT(i_1, i_2, 2) = (grad(2) + LUT(i_1, i_2, 2)) / 2;
                end

                im(y,x,:) = grad;
            end
        end
    end
end

display_gradient(im, 20);

% 
% % We must assume all images have the same size.
% [H, W, C] = size(ref1);
% % im = zeros(H,W,3);
% for y=1:H
%     for x=1:W
%         if ( sqrt((x-centerX)^2 + (y-centerY)^2) <= r )
% 
%             % Get the surface normal n as unit vec and ad we understand 
%             % ithe orientation w.r.t the sphere. 'z' should be -ve here. (?)
%             z = sqrt( r^2 - (x - centerX)^2 - (y - centerY)^2 );
%             n = [x; y; -z] - [centerX; centerY; 0];
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
% 
%             % lookup image intensities and make a hash map or something..
%             % ...
%             % EXPERIMENT CODE
% 
%             r1 = (double(ref1(y,x)) + 1) / (double(ref2(y,x)) + 1);
%             r2 = (double(ref2(y,x)) + 1) / (double(ref3(y,x)) + 1);
%             
% %             r1 = (r1 - (1 / 256)) / (256 - (1 / 256));
% %             r1 = r1^EXP;
% %             
% %             r2 = (r2 - (1 / 256)) / (256 - (1 / 256));
% %             r2 = r2^EXP;
%            
%             ratio_data1(:,i) = [double(ref2(y,x)); r1];
%             ratio_data2(:,i) = [double(ref3(y,x)); r2];
%             
%             %X = round(r1 * (LUT_SZ - 1)) + 1;
%             %Y = round(r2 * (LUT_SZ - 1)) + 1;
%             
%             %LUT(X, Y, 1) = grad(1);
%             %LUT(X, Y, 2) = grad(2);
%             
%             i = i + 1;
%             
%             % END EXPERIMENT CODE
%         end
%     end
% end
% 
% %histfit(ratio_data1(2,:))
% figure
% plot(ratio_data1(1,:),ratio_data1(2,:), '*')
% figure
% plot(ratio_data2(1,:),ratio_data2(2,:), '*')

[H, W, C] = size(ref1);
im = zeros(H,W,3);
for y=1:H
    for x=1:W
        if ( sqrt((x-centerX)^2 + (y-centerY)^2) <= r )

            % Get the intensity ratios then look up the corresponding
            % gradient vector in the LUT.
            r1 = (double(ref1(y,x)) + 1) / (double(ref2(y,x)) + 1);
            r2 = (double(ref2(y,x)) + 1) / (double(ref3(y,x)) + 1);
            
            % Get the indices of the pixel intensity ratios.
            [approx_r1, i_1] = min(abs(PERCENTILES - r1));
            [approx_r2, i_2] = min(abs(PERCENTILES - r2));
            PERCENTILES(i_1);
            PERCENTILES(i_2);
           
            grad = [LUT(i_1, i_2, 1), LUT(i_1, i_2, 2), -1];
            
%             while (grad == [0 0 -1])
%                 i_1 = i_1 - 1;
%                 i_2 = i_2 - 1;
%                 grad = [LUT(i_1, i_2, 1), LUT(i_1, i_2, 2), -1];
%             end
            im(y,x,:) = grad;
            
            % END EXPERIMENT CODE
        end
    end
end

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
function percentiles = generate_indices(im1, im2, im3, centerX, centerY, r, num_entries)

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
        s = [avg_x; avg_y; z] - [centerX; centerY; centerZ];
        s_i(i,:) = s ./ norm(s);
    end
end





