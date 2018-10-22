close all;

% ----------------------
% Weird things to remember to look at later:
%
% You have the loop currently constructing the lookup table from pixels
% close starting at y = 480 down to y = 1
%
% If p, q is 0, should also probably set the z to 0 when we lookup for
% surface normals.
%
% Playing with LUT now very much affects the output.. right now it is
% low. - too many entries could give more accurate but a loss of results!
% Note - 500 Gives representation of the hexagon! (any higher we lose it.)
%
% In display, you've normalized your gradients to have unit length
% (as unit normals)
% ----------------------

ref1 = rgb2gray(im2double(imread('./synth/sphere1R.png')));
ref2 = rgb2gray(im2double(imread('./synth/sphere2R.png')));
ref3 = rgb2gray(im2double(imread('./synth/sphere3R.png')));

%unknown1 = rgb2gray(im2double(imread('./synth/cone1.tif')));
%unknown2 = rgb2gray(im2double(imread('./synth/cone2.tif')));
%unknown3 = rgb2gray(im2double(imread('./synth/cone3.tif')));

unknown1 = rgb2gray(im2double(imread('./real/sphere-lamp1.tif')));
unknown2 = rgb2gray(im2double(imread('./real/sphere-lamp3.tif')));
unknown3 = rgb2gray(im2double(imread('./real/sphere-lamp2.tif')));

% imshow(unknown1);
% imshow(unknown2);
% imshow(unknown3);

[H, W, C] = size(ref1);
ref_imgs = {ref1, ref2, ref3};

% These params are for the non-resized sphere roughly:
% centerX = 320;
% centerY = 240;
% centerZ = 0;
% r       = 152;

% sphereXR.tif:
centerX = 360;
centerY = 260;
centerZ = 0;
r       = 152;

% imshow(ref1);
% imshow(ref2);
% imshow(ref3);
% imshow(ref1);

ratio_data1 = zeros(2,1);
ratio_data2 = zeros(2,1);
i = 1;

LUT_SZ = 5000;
CREATE_TABLE = false;

% ----------------------

s_i = get_s_vectors(centerX, centerY, centerZ, r, ref_imgs);

% ----------------------



% -----------------------

[unkn_H, unkn_W, unkn_C] = size(unknown1);
normal_im = zeros(unkn_H, unkn_W, 3);
grad_im   = zeros(unkn_H, unkn_W, 3);
for y=1:unkn_H
    for x=1:unkn_W
        % If all pixels are 0, then skip this pixel.
        if (unknown1(y,x) > 0 && unknown2(y,x) > 0 && unknown3(y,x) > 0)
            % Get the intensity ratios then look up the corresponding
            % gradient vector in the LUT.
            r1 = (double(unknown1(y,x))*255 + 1) / (double(unknown2(y,x))*255 + 1);
            r2 = (double(unknown2(y,x))*255 + 1) / (double(unknown3(y,x))*255 + 1);
            
            % Get the indices of the pixel intensity ratios.
            [approx_r1, i_1] = min(abs(PERCENTILES - r1));
            [approx_r2, i_2] = min(abs(PERCENTILES - r2));
            PERCENTILES(i_1);
            PERCENTILES(i_2);
           
            grad = [LUT(i_1, i_2, 1), -LUT(i_1, i_2, 2), 1];

            % ---------------------------------------
            
            if (grad(1,1) == 0 && grad(1,2) == 0)
                % Find the nearest non-zero.
                distances = (nz_row_ind - i_1).^2 + (nz_col_ind - i_2).^2;
                
                [min_dist, new_idx] = min(distances);
                
                new_i_1 = nz_row_ind(new_idx);
                new_i_2 = nz_col_ind(new_idx);
                grad = [LUT(new_i_1, new_i_2, 1), -LUT(new_i_1, new_i_2, 2), 1];
            end
            
            % ---------------------------------------
            
            grad_im(y,x,:)   = grad;
            normal_im(y,x,:) = grad / norm(grad);
            
        end
    end
end

display_gradient(normal_im, 20);

imshow(normal_im);


% TESTTIGMG
grad_im = normal_im;

grad_im = gradient_to_intensity(grad_im(:,:,1), grad_im(:,:,2));

MAX = max(grad_im(:))
MIN = min(grad_im(:))

grad_im = (grad_im - MAX) ./ (MIN - MAX);
imshow(imresize(grad_im, 4));

[unknH, unknW, unknC] = size(grad_im);
surf(1:unknW, 1:unknH, grad_im)
%surfnorm(im);

% ------------------------------------------------------------------
% 
function res = display_gradient(grad, sz)
    grad = flip(grad,1);
    [h, w, c] = size(grad);
    avg_horiz = grad((1:sz:h-sz),:,:);
    for i=1:sz-1
        avg_horiz = avg_horiz + grad((1:sz:h-sz)+i,:,:);
    end
    res = avg_horiz(:,(1:sz:w-sz),:);
    for i=1:sz
        res = res + avg_horiz(:,(1:sz:w-sz)+i-1,:);
    end
    res = res ./ sz^2;
    figure
    quiver(res(:,:,1), res(:,:,2));
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
        [row,col] = find( img >= 1 );
        avg_y = round(sum(row) / size(row,1));
        avg_x = round(sum(col) / size(col,1));

        z = sqrt( r^2 - (avg_x - centerX)^2 - (avg_y - centerY)^2 );
        
        % FLIP because the image is upside-down.
%         avg_y = -avg_y;
        
        s = [avg_x; avg_y; z] - [centerX; centerY; centerZ];
        s_i(i,:) = s ./ norm(s);
    end
end


% ------------------------------------------------------------------
%
function img = gradient_to_intensity(p, q)

    RESIZE_TO_CONVERT = true;
    RS_RATIO = 0.25;
    
    if RESIZE_TO_CONVERT
        p = imresize(p, RS_RATIO);
        q = imresize(q, RS_RATIO);
    end
    
    [imh, imw, nb] = size(p);
    % Set up the im2var mapping matrix.
    im2var = zeros(imh, imw);
    im2var(1:imh*imw) = 1:imh*imw;
    
    % For each pixel we need two equations, and so A should have 
    % 2*(#pix(src)) + 1 rows, where the + 1 is for the boundary condition. 
    % We need #pix(src) columns since we need to consider every pixel in
    % src.
    A = sparse(4*imw*imh + 1, imw*imh);
    b = zeros(4*imw*imh + 1, 1);

    e = 0;
    
    for y=1:(imh)
        for x=1:(imw)

            % Objective 1
            if ( x ~= imw )
                e = e + 1;
                % check bounds - at rightmost cases we leave the gradient
                % at zero, and so we can just ignore.
                A(e, im2var(y,x+1)) =  1;
                A(e, im2var(y,x))   = -1;
                b(e)                = p(y,x); 
            end
                        
            % Objective 2
            if ( y ~= imh )
                e = e + 1;
                % check bounds - at bottommost cases we leave the gradient
                % at zero, and so we can just ignore.
                A(e, im2var(y+1,x)) =  1;
                A(e, im2var(y,x))   = -1;   
                b(e)                = q(y,x);
            end
            
            if ( x ~= 1 )
                e = e + 1;
                % check bounds - at leftmost cases we leave the gradient
                % at zero, and so we can just ignore.
                A(e, im2var(y,x-1)) =  1;
                A(e, im2var(y,x))   = -1;
                b(e)                = -p(y,x-1);
            end

            if ( y ~= 1 )
                e = e + 1;
                % check bounds - at topmost cases we leave the gradient
                % at zero, and so we can just ignore.
                A(e, im2var(y-1,x)) =  1;
                A(e, im2var(y,x))   = -1;   
                b(e)                = -q(y-1,x);
            end
        end
    end
   
    % Objective 3
    e = e + 1;
    A(e, im2var(1,1)) = 1;
    b(e) = 0;
   
    % Solve the system.
    v = A\b;
    
    % Copy the pixels from the solution vector back into the image matrix
    % which will be returned.
    img = zeros(imh, imw);
    
    ind = 1;
    for x=1:imw
       for y=1:imh
           img(y,x) = v(ind);
           ind = ind + 1;
       end
    end
    
end

% ------------------------------------------------------------------
% 
function [LUT, LUT_SZ, PERCENTILES] = generate_table(s_i, LUT_SZ)

    % We must assume all images have the same size.
    ratio_data = zeros(1,1);
    ratio_data_ind = 1;
    
    START = -75;
    END   =  75;
    INCR  =  0.05;
    
    P = (START:INCR:END);
    P_sz = size(P,2);
    Q = (START:INCR:END);
    Q_sz = size(Q,2);

    for i=1:P_sz
        p = -P(1,i);
        for j=1:Q_sz
            
            q = -Q(1,j);
            n = [p,q,1];
            n = n / norm(n);

            % Now we can compute the 'intensities' from the dot
            % products of the light intensities (s_i) and the supposed
            % surface normal (n).
            E_1 = dot(n, s_i(1,:));
            E_2 = dot(n, s_i(2,:));
            E_3 = dot(n, s_i(3,:));

            % maybe should check if E_1 > 0 too                        ?????
            if ( E_1 > 0 && E_2 > 0 && E_3 > 0 )
                r1 = ((E_1 * 255) + 1) / ((E_2 * 255) + 1);
                r2 = ((E_2 * 255) + 1) / ((E_3 * 255) + 1);

                ratio_data(ratio_data_ind) = r1;
                ratio_data(ratio_data_ind+1) = r2;

                ratio_data_ind = ratio_data_ind + 2;
            end
        end
    end
    
    prctile_x =(1:100/LUT_SZ:100);
    PERCENTILES = prctile(ratio_data, prctile_x);
    
    %plot(1:size(prctile_x,2), PERCENTILES, '*');
    
    LUT_SZ = size(PERCENTILES, 2);
    LUT = zeros(LUT_SZ, LUT_SZ, 2);
    NUM_ENTRIES = ones(LUT_SZ, LUT_SZ);
    
    %im = zeros(Q_sz,P_sz,3);
    for i=1:P_sz
        p = -P(1,i);
        for j=1:Q_sz
            
            q = -Q(1,j);
            n = [p,q,1];
            n = n / norm(n);

            % Now we can compute the 'intensities' from the dot
            % products of the light intensities (s_i) and the supposed
            % surface normal (n).
            E_1 = dot(n, s_i(1,:));
            E_2 = dot(n, s_i(2,:));
            E_3 = dot(n, s_i(3,:));

            if ( E_1 > 0 && E_2 > 0 && E_3 > 0 )
                
                r1 = ((E_1 * 255) + 1) / ((E_2 * 255) + 1);
                r2 = ((E_2 * 255) + 1) / ((E_3 * 255) + 1);
                
                [diff_r1, i_1] = min(abs(PERCENTILES - r1));
                [diff_r2, i_2] = min(abs(PERCENTILES - r2));
                PERCENTILES(i_1);
                PERCENTILES(i_2);

%                 % Store in the table.
%                 if ( LUT(i_1, i_2, 1) == 0 && LUT(i_1, i_2, 2) == 0 ) 
%                     LUT(i_1, i_2, 1) = p;
%                     LUT(i_1, i_2, 2) = q;
%                     NUM_ENTRIES(i_1, i_2) = NUM_ENTRIES(i_1, i_2) + 1;
%                 end
                
                LUT(i_1, i_2, 1) = LUT(i_1, i_2, 1) + p;
                LUT(i_1, i_2, 2) = LUT(i_1, i_2, 2) + q;
                NUM_ENTRIES(i_1, i_2) = NUM_ENTRIES(i_1, i_2) + 1;

               % im(j,i,:) = n;
            end
        end
    end
   
end

