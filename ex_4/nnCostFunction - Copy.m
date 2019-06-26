function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
Theta={Theta1,Theta2};

% Setup some useful variables
m = size(X, 1);
X = [ones(m, 1) X];
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

 L=3;

% generate a(l) and z(l)
a{1}= X;
for l=1:L-1
z{l+1}=a{l}*Theta{l}';
a{l+1}=sigmoid(z{l+1});
    if l+1<L
        a{l+1} = [ones(size(a{l+1},1), 1) a{l+1}];  %to prevent adding bias unit for last layer
    end
end




%one hot encoded by myself
lenght=num_labels;
y2=zeros(size(y,1),lenght);
for i=1:size(y,1)
    for j=1:size(y,2)
       y2(i,y(i,j))=1;
    end
end

% %cost function without regularization
% for i=1:m
%     for k=1:num_labels
%         j(i,k)=(1/m)*(-y2(i,k)*log(a{3}(i,k))-(1-y2(i,k))*log(1-a{3}(i,k)));
%         J=J+j(i,k);
%     end
% end



%cost function with regularization 
Jnotr=0; %initialize J 
J=0;

                                 
for i=1:m
    for k=1:num_labels
        jnotr(i,k)=(1/m)*(-y2(i,k)*log(a{3}(i,k))-(1-y2(i,k))*log(1-a{3}(i,k)));
        Jnotr=Jnotr+jnotr(i,k);
    end
end

Jr1=0;
Jr2=0;
%regularization with for 

% for jj=1:hidden_layer_size
%     for kk=2:input_layer_size+1
%         jr1(jj,kk)=(lambda/2*m)*((Theta{1}(jj,kk))^2);
%         Jr1=Jr1+jr1(jj,kk);
%     end
% end
% 
% for jj=1:num_labels
%     for kk=2:hidden_layer_size+1
%         jr2(jj,kk)=(lambda/2*m)*((Theta{2}(jj,kk))^2);
%         Jr2=Jr2+jr2(jj,kk);
%     end
% end

%regularization with matrix 
%sqr each elements
aa=sum(sum((Theta{1}(:,2:end)).*(Theta{1}(:,2:end))));
bb=sum(sum((Theta{2}(:,2:end)).*(Theta{2}(:,2:end))));


Jreg=(lambda/(2*m))*(aa+bb);
J=Jnotr+Jreg;       
    
        
        
                
       

 
 
%%%HELP https://stats.stackexchange.com/questions/294873/what-is-the-significance-of-the-delta-matrix-in-neural-network-backpropagation
    %%% ?l is a matrix, and the dimensions of this matrix (assuming a fully connected neural net, which is what I think the tutorial is covering) is:
    %   nrows = number of nodes in the next layer, and ncolumns in the previous layer.

 % delta for each layer except layer 1
    delta{3}=a{3}-y2;
    for l=L-1:-1:2
        delta{l}=(delta{l+1}*(Theta{l})).*a{l}.*(1-a{l});   
    end

    delta{2}=delta{2}(:,2:end);
    
    
    %big Delta initialization except for last later
S=[input_layer_size,hidden_layer_size,num_labels];
%          for l=L-1:-1:1
%              if l==L-1
%                  Delta{l}=zeros(S(l)+1, S(l+1));      % last layer does not need bias
%              else
%                  Delta{l}=zeros(S(l)+1,S(l+1)+1);     % next Deltas should be added +1 to dimentions 
%              end
%          end
    
      for l=L-1:-1:1
          Delta{l}=zeros(S(l)+1, S(l+1));
      end
      
         
         
                                   
    % Big Delta calculation
        for l=1:L-1
            Delta{l}=Delta{l}+(a{l}'*delta{l+1});
        end
   
      
    % gradient descent 
    for l=1:L-1
        for i=1:(size(Delta{l},1))
            for j=1:(size(Delta{l},2))
                if j==1
                   D{l}(i,j)=(1/m)*(Delta{l}(i,j)+lambda*Theta{l}(j,i));
                else
                   D{l}(i,j)=(1/m)*(Delta{l}(i,j));
                end
            end
        end
    end
 
    % save on wanted arrays
Theta1_grad = D{1};
Theta2_grad =D{2};
               
        





















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
