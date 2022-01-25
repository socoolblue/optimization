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
%   sigle point crossOvern                                      %
%===============================================================

function ch=Crossover(p1,p2)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
VarMin= [0.5 0.5 0.5 0.5 ];                % x1 (1~41),  x2 (1~85), x3 (1~2), x4 (1~3)
VarMax= [41.5 85.5 2.5 3.5 ];


AverVar=(VarMin+VarMax)/2 ;

    x1=p1.Position %%-AverVar; 
    x2=p2.Position %%-AverVar;
    
    alpha=rand(size(x1));
    
    y1=alpha.*x1+(1-alpha).*x2;
    y2=alpha.*x2+(1-alpha).*x1;
    
    %  y1=y1+AverVar;
    %  y2=y2+AverVar;

    
    for k = 1:4 %
    y1(k)=round(y1(k))
    y2(k)=round(y2(k))
    end  
    %for k = 3:numel(y1)
    %y1(k)=y1(k)
    %y2(k)=y2(k)
    %end  
    
    
    
    ch=CreateEmptyIndividuals(2);
  
    ch(1).Position=y1; 
    ch(2).Position=y2;  

end