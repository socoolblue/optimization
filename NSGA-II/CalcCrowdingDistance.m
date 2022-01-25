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
%        Crowding Distance for Niche Sharing                    %
%===============================================================

function pop=CalcCrowdingDistance(pop,F)

    nobj=numel(pop(1).Cost);

    nf=numel(F);
    for f=1:nf
        
        A=F{f};
        C=GetCosts(pop(A));
        D=zeros(size(C));
        for j=1:nobj
            
            [SCj k]=sort(C(j,:));    %in ascending order
            
            D(j,k(1))=inf;
            D(j,k(end))=inf;
            D(j,k(2:end-1))=(SCj(3:end)-SCj(1:end-2))/(SCj(end)-SCj(1));
        end
        
        D=sum(D,1);
        for i=1:numel(A)
            pop(A(i)).CrowdingDistance=D(i);
        end
        
    end

end