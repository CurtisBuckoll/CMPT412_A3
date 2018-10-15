ref1 = rgb2gray(imread('./synth/sphere1.tif'));
ref2 = rgb2gray(imread('./synth/sphere2.tif'));
ref3 = rgb2gray(imread('./synth/sphere3.tif'));

centerX = 320;
centerY = 240;
r       = 152;

% imshow(ref1);
% imshow(ref2);
% imshow(ref3);
% imshow(ref1);

% We must assume all images have the same size.
[H, W, C] = size(ref1);
im = zeros(H,W,3);
for y=1:H
   for x=1:W
       if ( sqrt((x-centerX)^2 + (y-centerY)^2) <= r )
           
           % Get the surface normal n as unit vec and ad we understand 
           % ithe orientation w.r.t the sphere. 'z' should be -ve here. (?)
           z = sqrt( r^2 - (x - centerX)^2 - (y - centerY)^2 );
           n = [x; y; -z] - [centerX; centerY; 0];
           n = n ./ norm(n);
           
           % If the z component is 0, add a small delta and renormalize.
           DELTA = 0.01;
           if ( n(3) == 0 )
               n = n + DELTA;
               n = n / norm(n);
               continue;
           end
           
           % Obtain the 'gradient' vector here: (p,q,-1).
           grad = n ./ -n(3);
           
           % lookup image intensities and make a hash map or something..
           % ...
           %
           
           im(y,x,:) = grad;
       end
   end
end

display_gradient(im, 20);

[xc, yc] = 
ref1(yc,xc,:) = [1; 0; 0];
imshow(ref1);





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


% % ------------------------------------------------------------------
% % 
% function res = display_gradients(grad)
%     [h, w, c] = size(grad);
%     avg_horiz = grad((1:5:h),:,:) + grad((1:5:h)+1,:,:) + grad((1:5:h)+2,:,:) + grad((1:5:h)+3,:,:) + grad((1:5:h)+4,:,:);
%     res = avg_horiz(:,(1:5:w),:) + avg_horiz(:,(1:5:w)+1,:) + avg_horiz(:,(1:5:w)+2,:) + avg_horiz(:,(1:5:w)+3,:) + avg_horiz(:,(1:5:w)+4,:);
%     res = res ./ 25;
%     figure
%     x = res(:,:,1);
%     y = res(:,:,2);
%     quiver(x, y);
% end