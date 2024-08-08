import matplotlib.pyplot as plt

# Example data
x = [1, 2, 3, 4, 5]
y = [2, 3, 5, 7, 11]

# Create a new figure and a set of subplots
fig, ax = plt.subplots()

# Plot the data
ax.plot(x, y)

# Annotate a point on the plot
# Correctly specify xytext as a tuple or list of two elements
ax.annotate('Point of Interest', xy=(2, 3), xytext=(1.5, 2.5),
            arrowprops=dict(facecolor='black', shrink=0.05),
            bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.5))

# Set labels and title
ax.set_xlabel('X-axis Label')
ax.set_ylabel('Y-axis Label')
ax.set_title('Plot with Annotation')

# Display the grid
ax.grid(True)

# Show the plot
plt.show()