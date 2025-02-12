
import trimesh
import glooey

import pyglet
import trimesh.viewer
from src.animal import AnimalStruct
from src.collision_obj import CollisionObj


class Viewer:

    """
    Example application that includes moving camera, scene and image update.
    """

    def __init__(self, animal: AnimalStruct, object: CollisionObj, frame_index):
        # create window with padding
        self.width, self.height = 960, 720
        window = self._create_window(width=self.width, height=self.height)

        gui = glooey.Gui(window)

        hbox = glooey.HBox()
        hbox.set_padding(5)

        # scene widget for changing camera location
        self.object = object
        self.animal = animal
        self.frame_range = self.object.get_frame_range()
        self.frame_index = max(frame_index, self.frame_range[0])

        #Generate a scene with the combined geometries
        scene = trimesh.Scene()
        obj_geom = self.object.generate_geometry(frame_idx=self.frame_index)
        animal_ray, animal_node = self.animal.generate_geometry(frame_idx=self.frame_index)

        # Replace the current scene with the scene created in the animal class
        scene.add_geometry(animal_ray, node_name="animal_ray",
                                             geom_name=str(self.frame_index) + "_animal_ray")
        scene.add_geometry(animal_node, node_name="animal_node",
                                             geom_name=str(self.frame_index) + "_animal_node")

        scene.add_geometry(obj_geom, node_name="object", geom_name=str(self.frame_index)+"_object")
        # scene.add_geometry(trimesh.creation.axis(origin_size=0.04, axis_length=0.4))
        self.scene_widget = trimesh.viewer.SceneWidget(scene)
        hbox.add(self.scene_widget)

        gui.add(hbox)

        pyglet.clock.schedule_interval(self.callback, 1.0 / 30)
        pyglet.app.run()

    def callback(self, dt):
        #Update the frame counter
        self.frame_index = (self.frame_index + 1) % self.frame_range[1]



        #Check if the frame is present in the dict of animal frames
        if self.animal.check_frame_exist(self.frame_index):
            # Remove existing geometry from the scene
            self.scene_widget.do_undraw()

            animal_list = [k for k in self.scene_widget.scene.geometry.keys() if "animal" in k]
            self.scene_widget.scene.delete_geometry(animal_list)

            animal_ray, animal_node = self.animal.generate_geometry(frame_idx=self.frame_index)

            #Replace the current scene with the scene created in the animal class
            self.scene_widget.scene.add_geometry(animal_ray, node_name="animal_ray", geom_name=str(self.frame_index)+"_animal_ray")
            self.scene_widget.scene.add_geometry(animal_node, node_name="animal_node", geom_name=str(self.frame_index)+"_animal_node")


        #Check if the frame is present in the dict of object frames
        if self.object.check_frame_exist(self.frame_index):
            # Remove existing geometry from the scene
            self.scene_widget.do_undraw()

            obj_list = [k for k in self.scene_widget.scene.geometry.keys() if "obj" in k]
            self.scene_widget.scene.delete_geometry(obj_list)

            #Get the geometry from the dict in the obj class
            obj_geom = self.object.generate_geometry(self.frame_index)
            self.scene_widget.scene.add_geometry(obj_geom, node_name="object", geom_name=str(self.frame_index)+"_object")

        #Redraw the new scene
        self.scene_widget.do_draw()



    def _create_window(self, width, height):
        try:
            config = pyglet.gl.Config(
                sample_buffers=1, samples=4, depth_size=24, double_buffer=True
            )
            window = pyglet.window.Window(config=config, width=width, height=height)
        except pyglet.window.NoSuchConfigException:
            config = pyglet.gl.Config(double_buffer=True)
            window = pyglet.window.Window(config=config, width=width, height=height)

        @window.event
        def on_key_press(symbol, modifiers):
            if modifiers == 0:
                if symbol == pyglet.window.key.Q:
                    window.close()

        return window