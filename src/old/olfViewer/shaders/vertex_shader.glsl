#version 430 

layout(location = 0) in vec3 vertex_position;
layout(location = 1) in vec3 vertex_colour;

uniform mat4 camera_view, camera_proj, model_matrix;

out vec3 colour;

void main(void)
{
	colour = vertex_colour;
	gl_Position = camera_proj * camera_view * model_matrix * vec4 (vertex_position, 1.0);
}

