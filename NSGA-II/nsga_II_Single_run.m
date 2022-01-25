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
%   NSGA-II execution  (Experimetal Evaluation of ObjF)         %
%   "it" >=2                                                    %
%===============================================================

clc;
clear;
close all;

npop=10;                                    % main population size
nvar=4;                                     % number of unknown variables
nobj=2;                                % number of objective functions


pc=0.8;                         % crossover ratio
nc=round(pc*npop/2)*2;          % number of parents (also offsprings)

pm=0.2;                         % mutation ratio
nm=round(pm*npop);              % number of mutants


% main execution
it = 2;  % >=2
         
         
         
         
  if it == 2
        former_it = it - 1;
        data1 = load(sprintf('%d.txt',former_it));  
        pop1=CreateEmptyIndividuals(npop);
    for i=1:npop
        pop1(i).Position=[data1(i,1:nvar)] ; 
        pop1(i).Cost= [data1(i,nvar+1: nvar+2)];   
    end
        pop=pop1;  
  else
        former_it = it - 2;
        data1 = load(sprintf('%d.txt',former_it));  
        pop1=CreateEmptyIndividuals(npop);
    for i=1:npop
        pop1(i).Position=[data1(i,1:nvar)] ; 
        pop1(i).Cost= [data1(i,nvar+1: nvar+2)];   
    end
        pop=pop1; 
        
        data23 = load(sprintf('%d.txt',it - 1));
        pop23=CreateEmptyIndividuals(nc+nm);
    for i=1:nc+nm
        pop23(i).Position=[data23(i,1:nvar)] ; 
        pop23(i).Cost= [data23(i,nvar+1: nvar+2)];
    end
           pop=[pop1
            pop23];
 end 
 
  
    [pop F]=NonDominatedSorting(pop);
    
    pop=CalcCrowdingDistance(pop,F);

    pop=SortPopulation(pop);

    pop=pop(1:npop);

    
    [pop F]=NonDominatedSorting(pop);
    
    pop=CalcCrowdingDistance(pop,F);
   
    disp(['Iteration ' num2str(it) ':  Number of 1st Front Individuals = ' num2str(numel(F{1}))]);
       
    PlotFronts(pop,F);
  
    
     if it > 2
       
   pitP=reshape([pop.Position],nvar,[]);
   pitC=reshape([pop.Cost],nobj,[]) ;
   
pitPC=[[pitP]' [pitC]']';
       fgP=fopen(sprintf('%d.txt',it-1),'w');
       fprintf(fgP, '%g %g %g %g %g %g \n', [pitPC]);
       fclose(fgP);
    
     end

    
    
    pop2=CreateEmptyIndividuals(nc);
    
    for k=1:nc/2
        [p1, t1]=Tournament(pop);   
        [p2, t2]=Tournament(pop);    

        ch=Crossover(p1,p2);
        
        pop2(2*k-1)=ch(1);
        pop2(2*k)=ch(2);
    end

    pop3=CreateEmptyIndividuals(nm);
    for k=1:nm
        p=Tournament(pop);
        
        q=Mutate_Auto_Iter(p);
        
        pop3(k)=q;
    end
    
  
    pop=[pop2
         pop3];

       pitP=[[pop.Position]'];
       pitC=[[pop.Cost]'];
    

       
       
  % save New Generations.        
  fg=fopen(sprintf('%d.txt',it),'w');
  fprintf(fg, '%g %g %g %g \n', [pitP]);
  fclose(fg);
  
  data23 = load(sprintf('%d.txt',it))

