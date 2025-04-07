import numpy as np
import trimesh
import glooey

import pyglet
import trimesh.viewer
from src.animal import AnimalStruct
from src.object import CollisionObj, CollisionObjTransform
import numpy as np


class MultiViewer:

    """
    Viewer for both the animal and the object in the collision frame, allows for lists of objects and animals
    """

    def __init__(self, animal_list: [AnimalStruct], object_list: list[CollisionObj | CollisionObjTransform], frame_index, auto_play: bool = True, draw_legend: bool = True):
        # create window with padding
        self.width, self.height = 960, 720
        window = self._create_window(width=self.width, height=self.height)

        gui = glooey.Gui(window)

        hbox = glooey.HBox()
        hbox.set_padding(5)

        # scene widget for changing camera location
        self.object_list = object_list
        self.animal_list = animal_list

        min_frame = min([inst.get_frame_range()[0] for inst in [*self.animal_list, *self.object_list]])
        max_frame = max([inst.get_frame_range()[1] for inst in [*self.animal_list, *self.object_list]])

        self._frame_range = (min_frame, max_frame)
        self._frame_index = max(frame_index, self._frame_range[0])

        #Generate a scene with the combined geometries
        scene = trimesh.Scene()

        scene, animal_frame_flag = self.update_animal(scene, self.animal_list, self._frame_index)
        scene = self.update_obj(scene, self.object_list, self._frame_index)

        self._scene_widget = trimesh.viewer.SceneWidget(scene)
        hbox.add(self._scene_widget)
        self._label = pyglet.text.Label("Frame: " + str(self._frame_index),
                                        x=window.width / 20,
                                        y=window.height / 20,
                                        color=[222, 100, 200, 255],
                                        anchor_x='left',
                                        anchor_y='bottom',
                                        font_size= window.height / 30)

        self._legend = []
        if draw_legend:
            self._update_legend(animal_frame_flag)
        # self.label.draw()
        gui.add(hbox)

        if auto_play:
            pyglet.clock.schedule_interval(self._callback, 1.0 / 30)


        self._scene_widget.do_draw()
        pyglet.app.run()

    @staticmethod
    def update_obj(scene: trimesh.Scene, obj_list: list[CollisionObj | CollisionObjTransform], frame_idx: int):

        for obj_id, obj in enumerate(obj_list):
            # Check if the frame is present in the dict of object frames
            if obj.check_frame_exist(frame_idx):
                obj_name_list = [k for k in scene.geometry.keys() if "object_"+str(obj_id) in k]
                scene.delete_geometry(obj_name_list)

                # Get the geometry from the dict in the obj class
                if obj.__class__.__name__ == "CollisionObj":
                    obj_geom = obj.generate_geometry(frame_idx)
                    scene.add_geometry(obj_geom, node_name="object"+str(obj_id), geom_name=str(frame_idx)+"_object_"+str(obj_id))
                if obj.__class__.__name__ == "CollisionObjTransform":
                    obj_geom = obj.obj_mesh()
                    tf = obj.generate_transform(frame_idx)
                    scene.add_geometry(obj_geom, node_name="object" + str(obj_id),
                                       geom_name=str(frame_idx) + "_object_" + str(obj_id), transform=tf)


        return scene

    @staticmethod
    def update_animal(scene: trimesh.Scene, animal_list: list[AnimalStruct], frame_idx: int):
        animal_frame_flag = []
        for animal in animal_list:
            flag = False
            # Check if the frame is present in the dict of object frames
            if animal.check_frame_exist(frame_idx):
                animal_name_list = [k for k in scene.geometry.keys() if "animal_" + animal.name in k]
                scene.delete_geometry(animal_name_list)

                animal_geom = animal.generate_geometry(frame_idx=frame_idx)
                if animal_geom is not None:
                    flag = True
                    animal_ray, animal_node = animal_geom

                    # Replace the current scene with the scene created in the animal class
                    scene.add_geometry(animal_ray, node_name="animal_ray",
                                       geom_name=str(frame_idx) + "_animal_" + animal.name + "_ray")
                    scene.add_geometry(animal_node, node_name="animal_node",
                                       geom_name=str(frame_idx) + "_animal_" + animal.name + "_node")

            animal_frame_flag.append(flag)

        return scene, animal_frame_flag

    def _update_legend(self, animal_frame_flag: list[bool]):
        self._legend = []
        origin = np.array([self.width / 20, self.height - (self.height / 10)])
        width = self.width / 5
        height = max(self.height / (2 * len(self.animal_list)), 12)
        for flag, animal in zip(animal_frame_flag, self.animal_list):
            if flag:
                legend = pyglet.text.Label(animal.name,
                                           x=origin[0],
                                           y=origin[1],
                                           color=tuple(animal.colour),
                                           anchor_x='left',
                                           anchor_y='bottom',
                                           height=height,
                                           width=width)
                self._legend.append(legend)
                origin -= np.array([0, height * 1.1])

    def _callback(self, dt):
        #Update the frame counter
        self._frame_index = max((self._frame_index + 1) % self._frame_range[1], self._frame_range[0])
        self._label.text = "Frame: " + str(self._frame_index)
        self._scene_widget.do_undraw()
        self._scene_widget.scene = self.update_obj(self._scene_widget.scene, self.object_list, self._frame_index)
        self._scene_widget.scene, animal_frame_flag = self.update_animal(self._scene_widget.scene, self.animal_list, self._frame_index)
        self._update_legend(animal_frame_flag)

        #Redraw the new scene
        self._scene_widget.do_draw()

    def _create_window(self, width, height):
        try:
            config = pyglet.gl.Config(
                sample_buffers=1, samples=4, depth_size=24, double_buffer=True
            )
            window = pyglet.window.Window(config=config, width=width, height=height)
        except pyglet.window.NoSuchConfigException:
            config = pyglet.gl.Config(double_buffer=True)
            window = pyglet.window.Window(config=config, width=width, height=height)

        event_loop = pyglet.app.EventLoop()

        @event_loop.event
        def on_window_close(window):
            event_loop.exit()

        @window.event
        def on_key_press(symbol, modifiers):
            if modifiers == 0:
                if symbol == pyglet.window.key.Q:
                    window.close()

        @window.event
        def on_draw():
            self._label.draw()
            for legend in self._legend:
                legend.draw()

        return window