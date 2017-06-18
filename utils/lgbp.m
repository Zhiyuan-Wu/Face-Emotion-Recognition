% function Feature = lgbp(I)
% [Eim,Oim,Aim]=spatialgabor(I,3,0,0.1,0.1,0);
% Aim = uint8(floor(Aim/max(max(Aim))*256));
% imshow(Aim);
%
% function I_filtered = gabor(im, sigma, angle, freq)

I = rgb2gray(imread('1.jpg'));
im =I;


theta = 90;
lambda = 2/1.4;
sigma = lambda;
gamma2 = 2;
psi = 0;
gamma = 1;

sigma_x = sigma;
sigma_y = sigma/gamma;

% Bounding box
nstds = 3;
xmax = max(abs(nstds*sigma_x*cos(theta)),abs(nstds*sigma_y*sin(theta)));
xmax = ceil(max(1,xmax));
ymax = max(abs(nstds*sigma_x*sin(theta)),abs(nstds*sigma_y*cos(theta)));
ymax = ceil(max(1,ymax));
xmin = -xmax; ymin = -ymax;
[x,y] = meshgrid(xmin:xmax,ymin:ymax);

% Rotation
x_theta=x*cos(theta)+y*sin(theta);
y_theta=-x*sin(theta)+y*cos(theta);

G= exp(-.5*(x_theta.^2/lambda^2+y_theta.^2/lambda^2)).*exp(i*pi/lambda*x_theta+psi);
%surf(x,y,real(G));
I_filtered = abs(filter2(G,im));

%I_filtered = uint8(floor(I_filtered/max(max(I_filtered))*256));
I_filtered = I_filtered.^(gamma2);
%I_filtered = uint8(floor(I_filtered/max(max(I_filtered))*256));
I_filtered = uint8(floor(I_filtered/mean(mean(I_filtered))*128)) ;%- uint8(floor(I_filtered/mean(mean(im))*128));

imshow(I_filtered)
