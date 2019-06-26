function [h,h2]=plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%
A=length(y);
for i=1:A
    if y(i)==1
        h=plot(X(i,1),X(i,2),'k+','LineWidth', 2, 'MarkerSize', 7);
    else
        h2=plot(X(i,1),X(i,2),'ko','MarkerFaceColor', 'y','MarkerSize', 7);

    end
end




xlabel('Exam 1 score');
ylabel('Exam 2 score');
legend([h;h2],'Amitted','Not admitted');




% =========================================================================



hold off;


end
