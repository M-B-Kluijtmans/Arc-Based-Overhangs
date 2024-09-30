"""
Created on Tue Mar 26 09:39:58 2024

@author: Bas Kluijtmans
cite arc overhangs code:
    https://github.com/stmcculloch/arc-overhang
"""

from shapely.geometry import Polygon, LineString
from shapely import affinity #Used to apply rotating transformations to the arcs
import numpy as np #For computation 
import matplotlib.pyplot as plt #Used for plotting the shapes in the overhangs
import util #Python script containing useful functions for arc overhang generation. From: https://github.com/stmcculloch/arc-overhang
import geopandas as gpd #Used for plotting geometric objects
import os #For writing files to computer

###########################################################
#Experiment parameters
A = 1 #Values for the small radius threshold (0.35-1)
B = 2 #Values for the feedrate (2-4)
C = 2 #Values for the minimum number of arcs in a set (2-10)
D = 0.75 #Values for the threshold distance to the boundary line (0.35-0.75)


#Printer parameters

design_name = 'Arc_Overhang_Test'
nozzle_temp = 210
bed_temp = 40
#print_speed_1 = 1000            #Print speed during normal printing
#print_speed_2 = 5               #Print speed during printing of overhangs (suggested at 5 mm/s)
fan_percent = 100               #Suggested to be on full blast
arc_e_multiplier = 1.05         #Multiplier for extrusion when printing arcs
feedrate = B                    
brim_width = 5
filament_d = 1.75               #Diameter of the filament

#Design parameters

EW = 0.35          #Extrusion width
EH = 0.4           #Extrusion height (layer height)
initial_z = EH*0.6 #Initial nozzle position set to 0.6x the extrusion height for good bed adhesion
overhang_height = 20 #Height of layers in mm printed before overhang starts
base_height = 0.5   #Height of base (brim etc.)

#Parameters of the overhang polygon
R_poly = 10 #Average radius of the random polygon
irregularity = 0.6 #Adds irregularity to the angle point distribution of the polygon. Should be between 0 and 1
spikiness = 0.6 #Adds irregularity to the radius point distribution of the polygon. Should be between 0 and 1
N_poly = 10 #Number of vertices of the polygon
x_axis = 100 #x centre coordinate of polygon
y_axis = 50 #y centre coordinate of polygon

#Arc/Polygon parameters.
N_circle = 40 #Number of points the circles consist of
R_max = 10 #Maximum circle radius
R_min = C*EW #Minimum circle radius
Threshold = D #Defines 'buffer' that the arcs leave around the base polygon
Min_Arcs = np.floor(R_min/EW) #Minimum number of arcs is integer of minimum circle radius/extrusion width

#Prepare and create output file and list of image names
output_file_names = ["output/arcs_simple.gcode", "output/arcs_intermediate.gcode", "output/arcs_complex.gcode"]
image_names = ["arcs_simple.png", "arcs_intermediate.png", "arcs_complex.png"]

for output_file_name, image_name in zip(output_file_names, image_names):
    with open(output_file_name, 'w') as gcode_file:
        gcode_file.write(""";Printing arc overhangs\n""")
        
    with open('input/start.gcode','r') as start_gcode, open(output_file_name, 'a') as gcode_file:
        for line in start_gcode:
            gcode_file.write(line)

    ###########################################################
    #Create figure to plot the arcs in the overhang in
    fig, ax = plt.subplots(1,2) #2 subplots for gcode preview and rainbow visualization
    ax[0].set_aspect('equal')
    ax[1].set_aspect('equal')
    ax[0].title.set_text('Gcode Preview')
    ax[1].title.set_text('Rainbow Visualization')
    
    
    ###########################################################
    # List of polygon points lists
    if output_file_name == "output/arcs_simple.gcode":
        poly_points = [(98.4, 59.1), (91.6, 54.5), (93.4, 43.8), (103.0, 37.7), (109.7, 46.8), (106.2, 54.7)]
    elif output_file_name == "output/arcs_intermediate.gcode":
        poly_points = [(92.1, 64.7), (91.3, 50.9), (97.7, 49.2), (94.9, 43.1), (102.7, 38.7), (110.3, 45.6), (111.7, 55.5), (100.6, 59.4)]
    elif output_file_name == "output/arcs_complex.gcode":
        poly_points = [(110.4, 52.4), (108.7, 59.2), (100.0, 60.9), (90.2, 61.8), (93.9, 52.8), (92.3, 50.1), (85.6, 43.3), (96.9, 46.4), (99.9, 30.0), (108.0, 42.4)]
        
    
    base_poly = Polygon(poly_points)
    
    
    
    #Identify the largest edge to use as starting point
    
    p1, p2 = util.longest_edge(base_poly)
    starting_line = LineString([p1,p2])
    
    #Use the polygon (except the starting line) as a boundary line
    boundary_line = LineString(util.get_boundary_line(base_poly, p1))
    
    ##########################################################
    #Setting up the plots of the polygons and the arcs
    
    #Plot base polygon
    base_poly_geoseries = gpd.GeoSeries(base_poly)
    base_poly_geoseries.plot(ax=ax[0], color='white', edgecolor='black', linewidth=1)
    base_poly_geoseries.plot(ax=ax[1], color='white', edgecolor='black', linewidth=1)
    
    #Plot starting line
    starting_line_geoseries = gpd.GeoSeries(starting_line)
    starting_line_geoseries.plot(ax=ax[0], color='lime', linewidth=2)
    
    
    ###########################################################
    
    #Generating the arcs
    
    #First arc
    starting_point, r_start, r_farthest = util.get_farthest_point(starting_line, boundary_line, base_poly)
    starting_circle_norot = util.create_circle(starting_point.x, starting_point.y, r_start, N_circle)
    starting_line_angle = np.arctan2((p2.y-p1.y),p2.x-p1.x)
    starting_circle = affinity.rotate(starting_circle_norot, starting_line_angle, origin  ='centroid', use_radians=True)
    starting_arc = starting_circle.intersection(base_poly)
    
    
    ###########
    #Generating starting tower
    curr_z = EH  # Height of first layer
    with open(output_file_name, 'a') as gcode_file:
        gcode_file.write(f"G0 X{'{0:.3f}'.format(starting_point.x)} Y{'{0:.3f}'.format(starting_point.y)} F500\n")
        gcode_file.write(f"G1 Z{'{0:.3f}'.format(curr_z)} F500\n")
        gcode_file.write(";Generating first layer\n")
        gcode_file.write("G1 E3.8\n")  # Unretract
        
    # Fill in circles from outside to inside
    while curr_z < base_height:
        starting_tower_r = r_start + brim_width  
        while starting_tower_r > EW*2:
            first_layer_circle = util.create_circle(starting_point.x, starting_point.y, starting_tower_r, N_circle)
            util.write_gcode(output_file_name, first_layer_circle, EW, EH, filament_d, 2, feedrate*5, close_loop=True)
            starting_tower_r -= EW*2
        
        curr_z += EH
        with open(output_file_name, 'a') as gcode_file:
            gcode_file.write(f"G1 Z{'{0:.3f}'.format(curr_z)} F500\n")
    
    with open(output_file_name, 'a') as gcode_file:
        gcode_file.write(f"G1 Z{'{0:.3f}'.format(curr_z)} F500\n")
        gcode_file.write(";Generating tower\n")
        gcode_file.write("M106 S255 ;Turn on fan to max power\n") 
        
    while curr_z < overhang_height:
        util.write_gcode(output_file_name, starting_line.buffer(EW), EW, EH, filament_d, 2, feedrate*5, close_loop=True)
        with open(output_file_name, 'a') as gcode_file:
            gcode_file.write(f"G1 Z{'{0:.3f}'.format(curr_z)} F500\n")
        curr_z += EH
    
    curr_z -= EH*2
    
    with open(output_file_name, 'a') as gcode_file:
            gcode_file.write(f"G1 Z{'{0:.3f}'.format(curr_z)} F500\n")
    
    
    #Initialization next arcs
    r = EW #First arc has radius of the extrusion width
    r_small_arc = A #Threshold for arcs considered small
    curr_arc = starting_arc #Current arc being processed
    
    #First series of arcs
    starting_point = util.move_toward_point(starting_point, affinity.rotate(p1, 90, LineString([p1, p2]).centroid), EW*0.5) #move the starting point to the rotated version of the starting line
    while r < r_start-Threshold: #As long as the radius of the arc is less than or equal to the starting radius - the distance to the base polygon
        #Create a circle and rotate to the starting line
        next_circle = Polygon(util.create_circle(starting_point.x, starting_point.y, r, N_circle))
        next_circle = affinity.rotate(next_circle, starting_line_angle, origin = 'centroid', use_radians=True)
        
        #Generating and plotting the arcs
        next_arc = util.create_arc(next_circle, base_poly, ax, depth=0)
        if not next_arc:
                r += EW
                continue
        curr_arc = Polygon(next_arc)
        
        #Slow down and reduce flow for all small arcs
        if r < r_small_arc:
            speed_modifier = 0.25
            e_modifier = 0.25
        else: 
            speed_modifier = 1
            e_modifier = 1
    
        # Write gcode to file
        util.write_gcode(output_file_name, next_arc, EW, EH, filament_d, arc_e_multiplier*e_modifier, feedrate*speed_modifier, close_loop=False)
        
        r += EW #Increase radius with extrusion width to move to next arc
    
    #Determine space left in the polygon
    empty_space = base_poly.difference(curr_arc)
    #Find point on the current arc that is farthest away from the boundary line
    next_point, longest_distance, point_on_poly = util.get_farthest_point(curr_arc, boundary_line, base_poly)
    
    
    #Build arcs on the current arc, if there is still space
    
    while longest_distance > Threshold + Min_Arcs*EW:
        next_arc, empty_space, image_names = util.arc_overhang(curr_arc, boundary_line, starting_line_angle,
                                                                            N_circle, empty_space, next_circle, Threshold, ax, fig, 1, image_names,
                                                                            R_max, Min_Arcs, EW, output_file_name, EH, filament_d,
                                                                            arc_e_multiplier, feedrate)
        #Determine next point for next arc
        next_arc, longest_distance, point_on_poly = util.get_farthest_point(curr_arc, boundary_line, empty_space)
    
    ##########################################################
    #Generate perimeter
    for i in range(100):
        first_ring = LineString(Polygon(boundary_line).buffer(-99*EW + EW*i).exterior.coords)
        first_ring = first_ring.intersection(empty_space)
        if first_ring.length <1e-9:
            continue
    
        if first_ring.geom_type == 'LineString':
            line = first_ring
            # plot starting line
            first_ring_geoseries = gpd.GeoSeries(line)
            first_ring_geoseries.plot(ax=ax[0], color='blue', edgecolor = 'blue', linewidth=1)
            util.write_gcode(output_file_name, line, EW, EH, filament_d, arc_e_multiplier, feedrate, False)
        else:
            for line in first_ring.geoms:
                # plot starting line
                first_ring_geoseries = gpd.GeoSeries(line)
                first_ring_geoseries.plot(ax=ax[0], color='blue', edgecolor = 'blue', linewidth=1)
                util.write_gcode(output_file_name, line, EW, EH, filament_d, arc_e_multiplier, feedrate, False)

##########################################################
#End code and create image
    plt.savefig("output/arcs", dpi=600)
    plt.show()

for output_file_name in output_file_names:
    with open('input/end.gcode','r') as end_gcode, open(output_file_name,'a') as gcode_file:
        for line in end_gcode:
            gcode_file.write(line)
    
    
   

    



