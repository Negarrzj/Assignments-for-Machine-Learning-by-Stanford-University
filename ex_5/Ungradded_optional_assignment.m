clear ; close all; clc
% Load from ex5data1: 
% You will have X, y, Xval, yval, Xtest, ytest in your environment
load ('ex5data1.mat');
m = size(X, 1);

p = 3 ;

% Map X onto Polynomial Features and Normalize
X_poly = polyFeatures(X, p);
[X_poly, mu, sigma] = featureNormalize(X_poly);  % Normalize
X_poly = [ones(m, 1), X_poly];                   % Add Ones

% Map X_poly_val and normalize (using mu and sigma)
X_poly_val = polyFeatures(Xval, p);
X_poly_val = bsxfun(@minus, X_poly_val, mu);
X_poly_val = bsxfun(@rdivide, X_poly_val, sigma);
X_poly_val = [ones(size(X_poly_val, 1), 1), X_poly_val];           % Add Ones

fprintf('Normalized Training Example 1:\n');
fprintf('  %f  \n', X_poly(1, :));

%n= size(Xval,1);

error_train = zeros(m, 1);
error_val   = zeros(m, 1);

error_train_s = zeros(m, 1);
error_val_s = zeros(m, 1);
maxItter=50;

%X=[ones(m, 1) X];
%Xval=[ones(n, 1) Xval];
for itter = 1:maxItter
    
        % Compute train/cross validation errors using training examples 
    [X_s,ind]= datasample(X_poly,12);
    y_s= y(ind);
    [Xval_s,ind2]= datasample(X_poly_val,12);
    yval_s=yval(ind2);
    for i=1:m

        [theta] = trainLinearReg(X_s(1:i,:), y_s(1:i), 0.01); %theta optimizer with lambda
        [J1(i), grad]=linearRegCostFunction(X_s(1:i, :), y_s(1:i), theta, 0); %error_train should be with lambda=0
        error_train_s(i)=J1(i); %store
        [J2(i), grad]=linearRegCostFunction(Xval_s, yval_s, theta, 0); %error_train should be with lambda=0 %compute Jval with Xval and yval againn and again while we are increasing sample training sets
        error_val_s(i)=J2(i);%store
    end
    error_train_i(itter)={error_train_s};
    error_val_i(itter)={error_val_s};
end


sum_t=zeros(1,m);
sum_v=zeros(1,m);
for i=1:m
    for j=1:maxItter
        
        sum_t(i)=error_train_i{j}(i)+sum_t(i);
        sum_v(i)=error_val_i{j}(i)+sum_v(i);
    end
end

for i=1:m
    error_train (i)=sum_t(i)./10 ;
    error_val(i)= sum_v(i)./10;
end

plot(1:m, error_train, 1:m, error_val);
title('Learning curve for linear regression')
legend('Train', 'Cross Validation')
xlabel('Number of training examples')
ylabel('Error')
axis([0 13 0 150])

fprintf('# Training Examples\tTrain Error\tCross Validation Error\n');
for i = 1:m
    fprintf('  \t%d\t\t%f\t%f\n', i, error_train(i), error_val(i));
end