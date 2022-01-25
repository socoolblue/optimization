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
%          Random Mutation                                      %
%===============================================================

function q=Mutate_Auto_Iter(p)


VarMin= [0.5 0.5 0.5 0.5];                % ex) x1 (1~),  x2 (1~), x3 (1~), x4 (1~)
VarMax= [41.5 85.5 2.5 3.5];   

     y=p.Position;
     
    j=randi([1 numel(y)]);
    
    y(j)=unifrnd(VarMin(j),VarMax(j),[1 1]);

    for k = 1:4
    y(k)=round(y(k));
    end

    q=CreateEmptyIndividuals();
    q.Position=y;  

end