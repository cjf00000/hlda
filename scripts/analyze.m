clear all;

cs = dlmread('cs_mc-1.summary');
csmc = dlmread('cs_mc30.summary');
pcs = dlmread('pcs_mc-1.summary');
pcsmc = dlmread('pcs_mc30.summary');
hldac = dlmread('hlda-c.summary');

hold off;
fig = figure(1);
plot(cs(:,1), cs(:,2), 'ro');
hold on;
plot(csmc(:,1), csmc(:,2), 'rx');


plot(pcs(:,1), pcs(:,2), 'go');
plot(pcsmc(:, 1), pcsmc(:,2), 'gx');

plot(hldac(:,1), hldac(:,2), 'b^');

ylim([1000, 3000]);

legend('cs', 'cs (mc)', 'pcs', 'pcs (mc)', 'hlda-c');

print(fig, 'quality', '-dpng');
