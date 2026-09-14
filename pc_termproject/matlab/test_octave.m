tic;
d = load('/root/.claude/uploads/0544bbbf-a4a5-500d-b67c-8cecae91e9bb/15485546-example_2_rod.mat');
R = pc_pwem_official(d.epsr);
fprintf('rod: TM gaps:\n'); disp(R.tm_gaps(1:min(3,end),:));
fprintf('rod: TE gaps:\n'); disp(R.te_gaps(1:min(3,end),:));
% band values at X (kx=pi, ky=0) and M for comparison with Python/digitised figure
iX = find(abs(R.K(:,1)-pi)<1e-9 & abs(R.K(:,2))<1e-9); iM = find(abs(R.K(:,1)-pi)<1e-9 & abs(R.K(:,2)-pi)<1e-9);
fprintf('X TM: %s\n', sprintf('%.4f ', R.w_tm(iX,1:8))); fprintf('X TE: %s\n', sprintf('%.4f ', R.w_te(iX,1:8)));
fprintf('M TM: %s\n', sprintf('%.4f ', R.w_tm(iM,1:8))); fprintf('M TE: %s\n', sprintf('%.4f ', R.w_te(iM,1:8)));
fprintf('elapsed %.1f s\n', toc);
d = load('/root/.claude/uploads/0544bbbf-a4a5-500d-b67c-8cecae91e9bb/615dea0b-example_1_random.mat');
R2 = pc_pwem_official(d.epsr);
fprintf('random: TM gaps:\n'); disp(R2.tm_gaps); fprintf('random: TE gaps:\n'); disp(R2.te_gaps);
fprintf('random: TM band extrema (max of bands 1-6): %s\n', sprintf('%.5f ', max(R2.w_tm(:,1:6))));
fprintf('random: TE band extrema (min of bands 1-6): %s\n', sprintf('%.5f ', min(R2.w_te(:,1:6))));
