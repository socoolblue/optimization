%===============================================================
%                                                               %
%  MATLAB Code for Stepwise Opt.                                %
%  Non-dominated Sorting Genetic Algorithm II (NSGA-II)         %
%                                                               %
%                                                               %
%  Sejong Univ. K.-S. Sohn                                      %
%                                                               %
%         e-Mail: kssohn@sejong.ac.kr                           %
%         M.P:  010-6253-5913                                   %
%                                                               %
%       Test Benchmark Function                                 %
%===============================================================


function f=FittnessFunction(x)

    n=numel(x);
    
    f=[0 0];

    f(1)= 45+sum((x-1).^2-10*cos(2*(x-1).*pi));
    f(2)= 50+sum(x.^2-10*cos(2*x.*pi));
    
end