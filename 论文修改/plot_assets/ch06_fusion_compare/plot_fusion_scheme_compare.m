function plot_fusion_scheme_compare()
% 绘制图6-4至图6-6：融合系统与基线系统总体对比

root_dir = fileparts(mfilename('fullpath'));
font_name = 'Songti SC';
csv_path = fullfile(root_dir, 'fusion_scheme_compare.csv');
tbl = readtable(csv_path, 'TextType', 'string');

if any(strcmp('scheme_desc', tbl.Properties.VariableNames))
    raw_labels = tbl.scheme_desc;
else
    raw_labels = tbl.scheme;
end

scheme_labels = wrap_labels(raw_labels);
legend_labels = translate_scheme_labels(raw_labels);

% 图6-4：PR-AUC 对比
fig1 = figure('Color', 'w', 'Position', [120, 120, 780, 520]);
ax1 = axes(fig1);
bar(ax1, tbl.pr_auc, 'FaceColor', [1 1 1], 'EdgeColor', [0 0 0], 'LineWidth', 1.2);
apply_axis_style(ax1, font_name);
xticks(ax1, 1:height(tbl));
xticklabels(ax1, scheme_labels);
xlabel(ax1, '验证方案', 'FontName', font_name, 'FontSize', 13);
ylabel(ax1, 'PR-AUC', 'FontName', font_name, 'FontSize', 13);
ylim(ax1, [0.82, 0.94]);
annotate_single_series(ax1, tbl.pr_auc, 0.004, '%.4f', font_name);
exportgraphics(fig1, fullfile(root_dir, 'fig6_4_fusion_accuracy.png'), 'Resolution', 220);
close(fig1);

% 图6-5：吞吐对比
fig2 = figure('Color', 'w', 'Position', [120, 120, 780, 520]);
ax2 = axes(fig2);
bar(ax2, tbl.qps, 'FaceColor', [0.82 0.82 0.82], 'EdgeColor', [0 0 0], 'LineWidth', 1.2);
apply_axis_style(ax2, font_name);
xticks(ax2, 1:height(tbl));
xticklabels(ax2, scheme_labels);
xlabel(ax2, '验证方案', 'FontName', font_name, 'FontSize', 13);
ylabel(ax2, '吞吐率（QPS）', 'FontName', font_name, 'FontSize', 13);
ylim(ax2, [0, 5200]);
annotate_single_series(ax2, tbl.qps, 140, '%.2f', font_name);
exportgraphics(fig2, fullfile(root_dir, 'fig6_5_fusion_qps.png'), 'Resolution', 220);
close(fig2);

% 图6-6：效果—性能折中
fig3 = figure('Color', 'w', 'Position', [120, 120, 860, 560]);
ax3 = axes(fig3);
hold(ax3, 'on');
markers = {'o', 's', '^', 'd'};
line_styles = {'-', '--', '-.', ':'};
for i = 1:height(tbl)
    plot(ax3, tbl.qps(i), tbl.pr_auc(i), ...
        'Color', 'k', ...
        'LineStyle', line_styles{i}, ...
        'Marker', markers{i}, ...
        'MarkerSize', 7.5, ...
        'MarkerFaceColor', 'w', ...
        'LineWidth', 1.4, ...
        'DisplayName', legend_labels{i});
    [xoff, yoff, halign] = tradeoff_label_offset(i);
    text(tbl.qps(i) + xoff, tbl.pr_auc(i) + yoff, sprintf('%.4f / %.2f', tbl.pr_auc(i), tbl.qps(i)), ...
        'FontName', font_name, 'FontSize', 9.6, 'HorizontalAlignment', halign);
end
apply_axis_style(ax3, font_name);
xlabel(ax3, '吞吐率（QPS）', 'FontName', font_name, 'FontSize', 13);
ylabel(ax3, 'PR-AUC', 'FontName', font_name, 'FontSize', 13);
xlim(ax3, [2500, 4800]);
ylim(ax3, [0.83, 0.936]);
legend(ax3, 'Location', 'northoutside', 'Orientation', 'horizontal', 'NumColumns', 2, 'Box', 'off', 'FontName', font_name);
exportgraphics(fig3, fullfile(root_dir, 'fig6_6_fusion_tradeoff.png'), 'Resolution', 220);
close(fig3);
end

function apply_axis_style(ax, font_name)
ax.FontName = font_name;
ax.FontSize = 12;
ax.LineWidth = 1.0;
ax.Box = 'off';
ax.TickDir = 'out';
ax.Layer = 'top';
ax.YGrid = 'on';
ax.GridColor = [0.72, 0.72, 0.72];
ax.GridAlpha = 0.35;
end

function annotate_single_series(ax, values, offset, format_spec, font_name)
for i = 1:numel(values)
    text(ax, i, values(i) + offset, sprintf(format_spec, values(i)), ...
        'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'bottom', ...
        'FontName', font_name, ...
        'FontSize', 9.8);
end
end

function [xoff, yoff, halign] = tradeoff_label_offset(index)
switch index
    case 1
        xoff = 70; yoff = 0.0009; halign = 'left';
    case 2
        xoff = -70; yoff = 0.0006; halign = 'right';
    case 3
        xoff = 70; yoff = 0.0009; halign = 'left';
    otherwise
        xoff = 70; yoff = 0.0009; halign = 'left';
end
end

function labels = wrap_labels(values)
labels = cell(size(values));
for i = 1:numel(values)
    switch string(values(i))
        case {"Full-scoring baseline", "Full scoring"}
            labels{i} = "PyTOD全量" + newline + "评分基线";
        case {"Recall-only screening", "Recall-only"}
            labels{i} = "仅召回" + newline + "筛选";
        case {"Initial fusion pipeline", "Initial fusion"}
            labels{i} = "初步融合" + newline + "方案";
        case {"Full fusion pipeline", "Full fusion"}
            labels{i} = "完整融合" + newline + "方案";
        otherwise
            labels{i} = char(values(i));
    end
end
end

function labels = translate_scheme_labels(values)
labels = cell(size(values));
for i = 1:numel(values)
    switch string(values(i))
        case {"Full-scoring baseline", "Full scoring"}
            labels{i} = 'PyTOD全量评分基线';
        case {"Recall-only screening", "Recall-only"}
            labels{i} = '仅召回筛选';
        case {"Initial fusion pipeline", "Initial fusion"}
            labels{i} = '初步融合方案';
        case {"Full fusion pipeline", "Full fusion"}
            labels{i} = '完整融合方案';
        otherwise
            labels{i} = char(values(i));
    end
end
end
