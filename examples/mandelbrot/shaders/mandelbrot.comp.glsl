layout (local_size_x = 16, local_size_y = 16, local_size_z = 1) in;
DECL_PUSH_CONSTANTS {
  u32vec2 resolution;
  ResourceId backbuffer;
  float time;
} push_constants;

const vec2 ZOOM_COORD = vec2(-0.544008, -0.54998);
const vec2 TRAP_COORD = vec2(0.6003,0.023005);

vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

vec2 mendelbrot(vec2 c) {
  float a = c.x*c.x - c.y*c.y;
  float b = 2*c.x*c.y;

  return vec2(a, b);
}

void main() {
  vec2 coord = gl_GlobalInvocationID.xy;
  if(coord.x > push_constants.resolution.x || coord.y > push_constants.resolution.y) {
    return;
  }

  vec2 ndc = coord / push_constants.resolution;
  vec2 c = vec2(ndc.x * 2.0 - 1.0, 1.0 - 2.0 * ndc.y);
  vec2 extent = vec2(push_constants.resolution);
  c.x *= (extent.x / extent.y);
  c /= push_constants.time * push_constants.time;

  c += ZOOM_COORD;

  vec3 color = vec3(0);
  vec2 z = vec2(0);
  float distance = 100000000;
  
  float MAX_ITER = 128;
  
  float n = 0;
  for(int i = 0; i < MAX_ITER; i++) {
    n = i;
    float a = z.x*z.x-z.y*z.y;
    float b = 2*z.x*z.y;
    z = vec2(a,b) + c;

    float r = a + b;
    distance = min(r, distance);
    if(length(z) > 4) {
      break;
    }
  }
  n = n + 1 - log(log2(length(z)));

  float np = (n / MAX_ITER);
  float hue = 1 - np;
  float saturation = 1.0;
  float value = np;
  
  color = hsv2rgb(vec3(hue, saturation, value));

  imageStore(
    get_storage_image(push_constants.backbuffer),
    ivec2(coord),
    vec4(color, 1.0)
  );
}
