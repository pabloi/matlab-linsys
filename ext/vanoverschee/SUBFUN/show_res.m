% Function that shows the results of an application
function show_res(tit,m,l,n_id,n_val,...
    er1,er2,er3,er4,er5,er6,er7,er8,er9,er10,...
    er11,er12,er13,er14,er15,er16,er17,er18,er19,er20,tt1,tt2,n)

if (mean(er1) > 1000);er1 = Inf;end
if (mean(er2) > 1000);er2 = Inf;end
if (mean(er3) > 1000);er3 = Inf;end
if (mean(er4) > 1000);er4 = Inf;end
if (mean(er5) > 1000);er5 = Inf;end
if (mean(er6) > 1000);er6 = Inf;end
if (mean(er7) > 1000);er7 = Inf;end
if (mean(er8) > 1000);er8 = Inf;end
if (mean(er9) > 1000);er9 = Inf;end
if (mean(er10) > 1000);er10 = Inf;end
if (mean(er11) > 1000);er11 = Inf;end
if (mean(er12) > 1000);er12 = Inf;end
if (mean(er13) > 1000);er13 = Inf;end
if (mean(er14) > 1000);er14 = Inf;end
if (mean(er15) > 1000);er15 = Inf;end
if (mean(er16) > 1000);er16 = Inf;end
if (mean(er17) > 1000);er17 = Inf;end
if (mean(er18) > 1000);er18 = Inf;end
if (mean(er19) > 1000);er19 = Inf;end
if (mean(er20) > 1000);er20 = Inf;end


clc
disp(' ')
disp(' ')
disp(['      ',tit,':'])
disp(' ')
disp('  ----------------------------------------------------------------------------')
disp('  ||  Function  ||  Pred E (ID) |  Pred E (Val) || Sim E (ID) | Sim E (Val) ||')
disp('  ----------------------------------------------------------------------------')
disp(sprintf('  ||  subid     ||  %7.1f     |  %7.1f      || %7.1f    | %7.1f     ||',mean(er1),mean(er2),mean(er3),mean(er4)))
disp(sprintf('  ||  com_alt   ||  %7.1f     |  %7.1f      || %7.1f    | %7.1f     ||',mean(er5),mean(er6),mean(er7),mean(er8)))
disp(sprintf('  ||  com_stat  ||  %7.1f     |  %7.1f      || %7.1f    | %7.1f     ||',mean(er9),mean(er10),mean(er11),mean(er12)))
disp('  ----------------------------------------------------------------------------')
if (mean(er13) == 0)
  disp(sprintf('  ||  pem       ||       -      |       -       ||      -     |      -      ||'))
  disp(sprintf('  ||  oe        ||       -      |       -       ||      -     |      -      ||'))
else
  disp(sprintf('  ||  pem       ||  %7.1f     |  %7.1f      || %7.1f    | %7.1f     ||',mean(er13),mean(er14),mean(er15),mean(er16)))
  disp(sprintf('  ||  oe        ||  %7.1f     |  %7.1f      || %7.1f    | %7.1f     ||',mean(er17),mean(er18),mean(er19),mean(er20)))
end
disp('  ----------------------------------------------------------------------------')
disp(' ')
disp(['     Number of inputs:                 ',num2str(m)]);
disp(['     Number of outputs:                ',num2str(l)]);
disp(['     System order:                     ',num2str(n)]);
disp(['     Number of ID data points:         ',num2str(n_id)]);
disp(['     Number of VAL data points:        ',num2str(n_val)]);
disp(['     Computation Subspace:             ',int2str(tt1),' sec'])
disp(['     Computation Prediction Error:     ',int2str(tt2),' sec'])


