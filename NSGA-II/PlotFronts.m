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
%   Plot Final Pareto Frots                                     %
%===============================================================

function PlotFronts(pop,F)

    nf=numel(F);
    c=GetCosts(pop);
    
    h=linspace(0,2/3,nf);
    
    costs=cell(1,nf);
    legends=cell(1,nf);
    
    for f=1:nf
        costs{f}=c(:,F{f});
        legends{f}=['Front ' num2str(f)];
        
        color=hsv2rgb([h(f) 1 1]);
        
        plot(costs{f}(1,:),costs{f}(2,:),'*','color',color);
        hold on;
        
    end
    
    legend(legends);
    legend('Location','NorthEastOutside');
    
    hold off;

end