%Eigenspectrum plot
f0 = readDMAT("../data/realistic_biceps_coarse/run1/48eigs.dmat");
f1 = readDMAT("../data/simple_muscle_tendon/run6785/48eigs.dmat");
f3 = readDMAT("../data/leg_with_foot/run1/48eigs.dmat");
f4 = readDMAT("../data/cartoon_skeleton/run5/48eigs.dmat");
f5 = readDMAT("../data/contact_test/run5/48eigs.dmat");
f6 = readDMAT("../../FastMusclesPaperFigs/MeatHand/run1/48eigs.dmat");
f7 = readDMAT("../../FastMusclesPaperFigs/SoftRobot/run1/48eigs.dmat");

%plot configs
it = 0;
for f = [f0, f1,  f2, f3, f4, f5, f6, f7]
    h = plot(fliplr(t),1./f);
    h.LineWidth = 5;
    h.Color = [82, 91, 210]/255;
    h.Marker = ".";
    h.MarkerSize = 10;
    h.MarkerFaceColor = [82, 91, 210]/255
    h.MarkerEdgeColor = [82, 91, 210]/255;
    hold on
    it = it+1;
end
fig = gcf;
fig.Units = "inches";
fig.PaperUnits = "inches";
fig.PaperPosition = [0 0 4 2];
fig.Position = [0 0 4 2];

a = gca;
a.XLim = [t0, t1];
a.YLim = [0, max(1./[f0; f1;f2;f3;f4;f5;f6;f7])];
a.GridColor = [0 0 0]/255;
a.Color = [229 229 229]/255;
a.XTick = [0, 10, 20, 30, 40];
a.YTick = [0,100,200,300];
a.YGrid = "On";
a.GridColor = [1 1 1];
a.GridAlpha = 1;
a.Box = "off";
ylabel("Eigenvalue");
xlabel("Eigenvalue #");
title ("Inverse Eigenspectrum of ");
a.FontSize = 20;
a.FontWeight = 'bold';