cs = fscanf(fopen('cs_500.txt'), '%f', [2, 500]);
is = fscanf(fopen('is_500.txt'), '%f', [2, 500]);

figure(1);
semilogx(cs(2,:));
hold on;
semilogx(is(2,:));
legend('Collapsed', 'Instantiated');
xlabel('Iteration');
ylabel('Perplexity');
print('-dpng', 'iter-per.png');

figure(2);
plot(cs(1,:));
hold on;
plot(is(1,:));
legend('Collapsed', 'Instantiated');
xlabel('Iteration');
ylabel('#topics');
print('-dpng', 'iter-topics.png');

close all;
