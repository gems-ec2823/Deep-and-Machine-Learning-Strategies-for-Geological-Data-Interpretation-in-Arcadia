import pandas as pd
import ipywidgets as widgets
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from matplotlib.image import imread
from ipywidgets import HBox, Output
import os

def handle_data():
    log1 = pd.read_csv('csv_wireline_logs/well_204_19_3A.csv')
    log2 = pd.read_csv('csv_wireline_logs/well_204_19_6.csv')
    log3 = pd.read_csv('csv_wireline_logs/well_204_19_7.csv')
    log4 = pd.read_csv('csv_wireline_logs/well_204_20_1.csv')
    log5 = pd.read_csv('csv_wireline_logs/well_204_20_1Z.csv')
    log6 = pd.read_csv('csv_wireline_logs/well_204_20_2.csv')
    log7 = pd.read_csv('csv_wireline_logs/well_204_20_3.csv')
    log8 = pd.read_csv('csv_wireline_logs/well_204_20_6a.csv')
    log9 = pd.read_csv('csv_wireline_logs/well_204_20a_7.csv')
    log10 = pd.read_csv('csv_wireline_logs/well_204_24a_6.csv')
    log11 = pd.read_csv('csv_wireline_logs/well_204_24a_7.csv')

    fa1 = pd.read_csv('Data/labels/faciesclass/204-19-3a.csv')
    fa2 = pd.read_csv('Data/labels/faciesclass/204-19-6.csv')
    fa3= pd.read_csv('Data/labels/faciesclass/204-19-7.csv')
    fa4 = pd.read_csv('Data/labels/faciesclass/204-20-1.csv')
    fa5 = pd.read_csv('Data/labels/faciesclass/204-20-1Z.csv')
    fa6 = pd.read_csv('Data/labels/faciesclass/204-20-2.csv')
    fa7 = pd.read_csv('Data/labels/faciesclass/204-20-3.csv')
    fa8 = pd.read_csv('Data/labels/faciesclass/204-20-6a.csv')
    fa9 = pd.read_csv('Data/labels/faciesclass/204-20a-7.csv')
    fa10 = pd.read_csv('Data/labels/faciesclass/204-24a-6.csv')
    fa11 = pd.read_csv('Data/labels/faciesclass/204-24a-7.csv')

    def add_and_filter_facies(lg, df):
        """
        Adds 'Facies Class Name' to the log DataFrame based on the depth ranges in df and removes rows where 'Facies Class Name' is NaN.

        Parameters:
        log (DataFrame): The DataFrame containing the DEPTH column.
        df (DataFrame): The DataFrame containing Start Depth, End Depth, and Facies Class Name columns.

        Returns:
        DataFrame: The modified log DataFrame with the 'Facies Class Name' column added and NaN rows removed.
        """

        def map_facies_class(depth, df):
            facies_class = df[(df['Start Depth'] <= depth) & (df['End Depth'] >= depth)]['Facies Class Name']
            if not facies_class.empty:
                return facies_class.iloc[0]
            else:
                return None

        lg['Facies Class Name'] = lg['DEPTH'].apply(lambda depth: map_facies_class(depth, df))
        lg = lg.dropna(subset=['Facies Class Name'])

        return lg

    well1 = add_and_filter_facies(log1, fa1)
    well2 = add_and_filter_facies(log2, fa2)
    well3 = add_and_filter_facies(log3, fa3)
    well4 = add_and_filter_facies(log4, fa4)
    well5 = add_and_filter_facies(log5, fa5)
    well6 = add_and_filter_facies(log6, fa6)
    well7 = add_and_filter_facies(log7, fa7)
    well8 = add_and_filter_facies(log8, fa8)
    well9 = add_and_filter_facies(log9, fa9)
    well10 = add_and_filter_facies(log10, fa10)
    well11 = add_and_filter_facies(log11, fa11)

    # make directory if it does not exist
    if not os.path.exists('handle_data'):
        os.makedirs('handle_data')

    well1.to_csv('handle_data/204_19_3A.csv', index=False)
    well2.to_csv('handle_data/204_19_6.csv', index=False)
    well3.to_csv('handle_data/204_19_7.csv', index=False)
    well4.to_csv('handle_data/204_20_1.csv', index=False)
    well5.to_csv('handle_data/204_20_1Z.csv', index=False)
    well6.to_csv('handle_data/204_20_2.csv', index=False)
    well7.to_csv('handle_data/204_20_3.csv', index=False)
    well8.to_csv('handle_data/204_20_6a.csv', index=False)
    well9.to_csv('handle_data/204_20a_7.csv', index=False)
    well10.to_csv('handle_data/204_24a_6.csv', index=False)
    well11.to_csv('handle_data/204_24a_7.csv', index=False)

def visualize():
    # Global constants
    facies_colors = {
        's': (255, 0, 0, 128),   # red with transparency
        'sh': (0, 255, 0, 128),  # green with transparency
        'nc ': (0, 0, 0, 128),    # black with transparency
        'ih and is': (0, 0, 255, 128),  # blue with transparency
        'os': (255, 255, 0, 128) # yellow with transparency
    }
    
    # Convert the RGBA colors to hexadecimal colors and ignore the alpha channel
    FACIES_COLORS = ['#{:02X}{:02X}{:02X}'.format(r, g, b) for r, g, b, a in facies_colors.values()]

    IMAGE_FILE= 'Data/204_20_1Z.png'

    def handle_data():
        pass

    def process_permeability_data():
        permeability = {}
        wells = [ '204_19_7', '204_20_1', '204_20_1Z', '204_20_2', '204_20_3', '204_20_6a', '204_20a_7', '204_24a_6', '204_24a_7']
        for well in wells:
            file_path = f'Data/labels/permeability/{well}.csv'
            try:
                df = pd.read_csv(file_path)

                for column in df.columns:
                    df[column] = pd.to_numeric(df[column], errors='coerce')

                permeability[well] = df
            except FileNotFoundError:
                print(f"WELL DATA NOT AVAILABLE {well}")
                permeability[well] = pd.DataFrame()  #
        return permeability

    permeability= process_permeability_data()

    def create_density_and_select_columns(df, columns_to_keep):
        """
        Calculate density by adding DENC and DENS columns and filter the DataFrame
        based on specified columns.

        Parameters:
        df (pandas.DataFrame): DataFrame containing well data.
        columns_to_keep (list): List of columns to retain in the DataFrame.

        Returns:
        pandas.DataFrame: Processed DataFrame with calculated density and filtered columns.
        """
        df['density'] = pd.to_numeric(df['DENC'], errors='coerce') + pd.to_numeric(df['DENS'], errors='coerce')
        columns_to_keep = [col for col in columns_to_keep if col in df.columns] + ['density']
        return df[columns_to_keep]

    def create_depth_and_select_columns(df, columns_to_keep):
        """
        Filter the DataFrame based on specified columns including DEPTH and PERMEABILITY.

        Parameters:
        df (pandas.DataFrame): DataFrame containing well data.
        columns_to_keep (list): List of columns to retain in the DataFrame.

        Returns:
        pandas.DataFrame: Filtered DataFrame based on specified columns.
        """
        columns_to_keep = [col for col in columns_to_keep if col in df.columns] + ['DEPTH', 'PERMEABILITY (HORIZONTAL)\nKair\nmd']
        return df[columns_to_keep]

    def facies_plot(logs, facies_colors, image_file, permeability):
        # Sorting and colormap setup
        logs = logs.sort_values(by='DEPTH')
        cmap_facies = ListedColormap(facies_colors)
        ztop, zbot = logs['DEPTH'].min(), logs['DEPTH'].max()

        # available data
        permeability_available = not permeability.empty

        # Determine columns to plot
        columns_to_plot = [col for col in logs.columns if col not in ['DEPTH', 'Facies Class Name']]
        num_subplots = len(columns_to_plot) + (2 if permeability_available else 1)
        line_colors = plt.cm.viridis(np.linspace(0, 1, len(columns_to_plot)))

        # Plot setup
        f, ax = plt.subplots(nrows=1, ncols=num_subplots, figsize=(num_subplots*2.5 , 10))
        for i, col in enumerate(columns_to_plot):
            ax[i].plot(logs[col], logs['DEPTH'], '-', color=line_colors[i])
            ax[i].set_xlabel(col)
            ax[i].set_ylim(ztop, zbot)
            ax[i].invert_yaxis()
            ax[i].grid()
            if i > 0:
                ax[i].set_yticklabels([])

        if permeability_available:
            permeability_filtered = permeability[(permeability['DEPTH'] >= ztop) & (permeability['DEPTH'] <= zbot)]
            permeability_filtered['PERMEABILITY (HORIZONTAL)\nKair\nmd'] = permeability_filtered['PERMEABILITY (HORIZONTAL)\nKair\nmd'].fillna(0)
            ax[-3].barh(permeability_filtered['DEPTH'], permeability_filtered['PERMEABILITY (HORIZONTAL)\nKair\nmd'], height=0.2, color='blue')
            ax[-3].set_xlabel('Permeability')
            ax[-3].set_ylim(ztop, zbot)
            ax[-3].invert_yaxis()
            ax[-3].grid()


        else:
            pass

        facies_values = pd.factorize(logs['Facies Class Name'])[0]
        cluster = np.repeat(np.expand_dims(facies_values, 1), 100, 1)
        im = ax[-1].imshow(cluster, interpolation='none', aspect='auto', cmap=cmap_facies)
        ax[-1].set_xlabel('Facies')
        ax_image = ax[-2]  
        img = imread(image_file)
        ax_image.imshow(img, aspect='auto')
        ax_image.axis('off')

        ax_image.set_aspect(0.1, adjustable='box')

        # Colorbar setup
        divider = make_axes_locatable(ax[-1])
        cax = divider.append_axes("right", size="10%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label('Facies Classes')
        unique_facies = logs['Facies Class Name'].unique()
        tick_locations = np.arange(len(unique_facies))
        cbar.set_ticks(tick_locations)
        cbar.set_ticklabels(unique_facies)

        plt.tight_layout()
        plt.show()

    def process_well_data():
        """
        Process and load well data from CSV files.

        Returns:
        dict: Dictionary containing well data as pandas DataFrames.
        """
        columns_to_keep = ['CALI', 'DTC', 'GR', 'PEF', 'RDEP', 'RMIC','NEUT','DEPTH','Facies Class Name']
        wells = ['204_19_3A', '204_19_6', '204_19_7', '204_20_1', '204_20_1Z', '204_20_2', '204_20_3', '204_20_6a', '204_20a_7', '204_24a_6', '204_24a_7']
        well_data = {
            well_name: {
                "data": create_density_and_select_columns(pd.read_csv(f'handle_data/{well_name}.csv'), columns_to_keep),
                "image": f'Data/labels/{well_name}.png' 
            } for well_name in wells
        }


        return well_data


    def create_well_selector(well_data):
        """
        Create a dropdown widget for well selection.

        Parameters:
        well_data (dict): Dictionary containing well data.

        Returns:
        ipywidgets.Dropdown: Dropdown widget for well selection.
        """
        return widgets.Dropdown(options=list(well_data.keys()), description='Select Well:')


    def on_well_selected(change, well_data, facies_colors, image_file, well_output):
        """
        Handle well selection event and display the facies plot.

        Parameters:
        change (dict): Dictionary containing change information.
        well_data (dict): Dictionary containing well data.
        facies_colors (list): List of colors for facies classification.
        image_file (str): Path to the well image file.
        """
        clear_output(wait=True)
        display(well_selector)
        selected_well = change['new']
        selected_well_data = well_data[selected_well]["data"]
        selected_well_image = well_data[selected_well]["image"]
        selected_well_permeability = permeability.get(selected_well, pd.DataFrame())
        facies_plot(selected_well_data, facies_colors, selected_well_image, selected_well_permeability)


    # Main execution

    well_data = process_well_data()
    well_selector = create_well_selector(well_data)
    well_output = Output()
    well_selector.observe(lambda change: on_well_selected(change, well_data, FACIES_COLORS, IMAGE_FILE, well_output), names='value')
    
    widgets_container = HBox([well_selector])

    display(widgets_container)



def predict_visualize(log, Permeability, original_image, predicted_image):
    # Assuming the depth order in log and Permeability array is the same
    # Add the Permeability array as a new column in the log DataFrame
    log['predict_Permeability'] = Permeability

    # Sorting log by 'DEPTH'
    log = log.sort_values(by='DEPTH')
    ztop, zbot = log['DEPTH'].min(), log['DEPTH'].max()

    # Columns from log to be plotted
    log_columns = [ 'DENS', 'DTC', 'GR', 'PEF', 'RDEP', 'predict_Permeability']

    # Setup the number of subplots
    num_log_subplots = len(log_columns)
    total_subplots = num_log_subplots + 2  # Adding 2 for original image and predicted image

    # Plot setup
    f, ax = plt.subplots(nrows=1, ncols=total_subplots, figsize=(total_subplots * 2, 10))

    # Plot each log column
    for i, col in enumerate(log_columns):
        ax[i].plot(log[col], log['DEPTH'], '-')
        ax[i].set_xlabel(col)
        ax[i].set_ylim(ztop, zbot)
        ax[i].invert_yaxis()
        ax[i].grid()
        if i > 0:
            ax[i].set_yticklabels([])

    # Plot for original image
    ax_image_original = ax[num_log_subplots]
    ax_image_original.imshow(original_image, aspect='auto')
    ax_image_original.axis('off')

    # Plot for predicted image
    ax_image_predicted = ax[num_log_subplots + 1]
    ax_image_predicted.imshow(predicted_image, aspect='auto')
    ax_image_predicted.axis('off')

    display(ax_image_predicted)
    


