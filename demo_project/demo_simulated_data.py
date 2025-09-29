import threading
import queue
import time
import random
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

from eeg_generator import EEGProducer, live_consumer, queue_stream

# New BCI functions for generated data
def simulate_training_data(n_samples=100, n_channel=4, sampling_rate=10):
    """Generate training data for simulated classes."""
    data = []
    labels = []
    t = np.linspace(0, 1, sampling_rate)  # 1 second window
    for _ in range(n_samples):
        cls = random.choice([0, 1])
        if cls == 0:  # Left
            sample = np.array([np.sin(2 * np.pi * 10 * t) + np.random.normal(0, 0.1, sampling_rate) for _ in range(n_channel)])
            sample[2:] = np.random.normal(0, 1, (2, sampling_rate))
        else:  # Right
            sample = np.array([np.sin(2 * np.pi * 12 * t) + np.random.normal(0, 0.1, sampling_rate) for _ in range(n_channel)])
            sample[:2] = np.random.normal(0, 1, (2, sampling_rate))
        features = np.var(sample, axis=1)  # Simple variance features per channel
        data.append(features)
        labels.append(cls)
    return np.array(data), np.array(labels)

def train_simulated_classifier():
    """Train a simple LDA on simulated features."""
    X, y = simulate_training_data()
    clf = LinearDiscriminantAnalysis()
    clf.fit(X, y)
    return clf

def online_classify_and_move(clf, out_queue, n_channel=4, sampling_rate=10, window_size=10):
    """Online classification from stream and move cube."""
    window = []
    
    # Initialize Pygame and OpenGL
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    gluPerspective(75, (display[0] / display[1]), 0.1, 50.0)
    glTranslatef(0.0, 0.0, -5)
    
    cube_position = 0.0
    
    while True:
        sample = out_queue.get()
        if sample is None:
            break
        window.append(sample)
        if len(window) > window_size:
            window.pop(0)
        if len(window) == window_size:
            data = np.array(window).T  # (n_channel, window_size)
            features = np.var(data, axis=1)  # Variance per channel
            pred = clf.predict([features])[0]
            if pred == 0:  # Left
                cube_position -= 0.2
            else:  # Right
                cube_position += 0.2
            
            # Render
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glTranslatef(cube_position - cube_position_prev if 'cube_position_prev' in locals() else 0, 0, 0)
            cube_position_prev = cube_position
            draw_cube()
            pygame.display.flip()
            pygame.time.wait(100)  # Update rate
    
    pygame.quit()

def draw_cube():
    """Draw a simple cube using OpenGL."""
    vertices = (
        (1, -1, -1), (1, 1, -1), (-1, 1, -1), (-1, -1, -1),
        (1, -1, 1), (1, 1, 1), (-1, -1, 1), (-1, 1, 1)
    )
    edges = (
        (0,1), (0,3), (0,4), (2,1), (2,3), (2,7),
        (6,3), (6,4), (6,7), (5,1), (5,4), (5,7)
    )
    glBegin(GL_LINES)
    for edge in edges:
        for vertex in edge:
            glVertex3fv(vertices[vertex])
    glEnd()

# Example usage for generated data
if __name__ == "__main__":
    q = queue.Queue()
    producer = EEGProducer(n_channel=4, sampling_rate=10, duration=10, out_queues=[q], simulate_class=None)  # Randomly choose class or noise
    producer.start()
    
    clf = train_simulated_classifier()
    online_classify_and_move(clf, q)
    
    producer.join()