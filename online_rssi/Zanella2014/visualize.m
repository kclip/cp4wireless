%%
num_ch =            16; % num of channels
v_ch_id =           10 + [1:num_ch]; % channel IDS
my_colormap =       hsv(num_ch);
col_ch_id =         3;
map_col_ch_id =     containers.Map({'Aisle','Desk','Lab','Outdoor','Room'},[4,3,4,4,3]);
col_rssi =          5;

for scenario = {'Aisle','Desk','Lab','Outdoor','Room'}
    load([scenario{1},'.mat']); % M
    v_time =        1:size(M,1);
    col_ch_id =     map_col_ch_id(scenario{1});
    figure; hold on; title(scenario{1});
    for i_ch_id = 1:num_ch
        ind_of_ch =         M(:,col_ch_id)==v_ch_id(i_ch_id);
        plot(v_time(ind_of_ch),M(ind_of_ch,col_rssi),'.','MarkerFaceColor',my_colormap(i_ch_id,:));
    end
end