#include "rtweekend.h"

#include "camera.h"
#include "hittable.h"
#include "hittable_list.h"
#include "sphere.h"

int main() {

    hittable_list world;

    double deltaX = 0.1;

    world.add(make_shared<sphere>(point3(0  -deltaX,0,-1)  , 0.5));
    world.add(make_shared<sphere>(point3(-1 -deltaX,0,-1) , 0.5));
    world.add(make_shared<sphere>(point3(-2 -deltaX,0,-1) , 0.5));
    world.add(make_shared<sphere>(point3(1.65-deltaX,0.3,-1), 0.5));
    world.add(make_shared<sphere>(point3(22,-199.7,-1), 200));

    camera cam;

    cam.center = point3(0,0,51.2);
    cam.focal_length = 20;

    cam.aspect_ratio = 1.0;
    cam.image_width  = 400;
    cam.samples_per_pixel = 50;

    cam.render(world);

}