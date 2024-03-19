// SOLVE THE ACOUSTIC EQUATION ON AN HETEROGENEOUS DOMAIN
// rho*d²p/dt² = div(k.grad(p))
const {
    GPUComposer,
    GPULayer,
    GPUProgram,
    renderAmplitudeProgram,
    renderSignedAmplitudeProgram,
    renderRGBProgram,
    addLayersprogram,
    copyProgram,
    FLOAT,
    INT,
    REPEAT,
    CLAMP_TO_EDGE,
    NEAREST,
    LINEAR,
} = GPUIO ;

// Init a canvas element.
	const canvas = document.createElement('canvas');
	document.body.appendChild(canvas);

// DEFAULT PARAMETERS
const PARAMS = {
  Engine: {
    StepsByFrame: 1, // number of stepping StepsByFrame
    PressureScale: 0.5, // pressure scaling
  },
  Material: {
    rho: .5, // initial material density
    c: .5, // nitial wave velocity
    eta_log: math.log10(.001), // initial material damping
    painting: false,
    strength: .5, // paint strength
  },
  Rain: {
    Active: false, // is it rainin ?
    DropDiameter: {min:5.0, max:20.0}, // source radius in pixels
    DropPeriod: {min:5, max:15}, // drop period in millisecs
  },
  theme: 'dark',
}
// Material 2 colors using YUV convention
function mat2clr(mat) {
// Convert material properties to RGBA color
  let Y=1-mat.rho; let U=mat.c-.5; let V=mat.eta_log/6+.5;
  let R=Y+V; let G=(Y-(U+V)/2); let B=Y+U;
  return [R,G,B,1.0]
}
function clr2mat(rgb) {
// Convert RGBA color back to material props
  let R=rgb[0]; let G=rgb[1]; let B=rgb[2];
  let Y=(R+2*G+B)/4; let U=B-Y; let V=R-Y;
  let rho=1-Y; let c=U+.5; let eta_log=6*V-3;
  return [rho,c,eta_log,1.0]
}

// Init a simple gui. (see https://cocopon.github.io/tweakpane/quick-tour/)
const pane = new Tweakpane.Pane({title:'2D Heterogeneous Wave'});
pane.registerPlugin(TweakpaneEssentialsPlugin);
const fpsGraph = pane.addBlade({view: 'fpsgraph',label: 'fpsgraph',});
pane.addButton({title:"Reset Pressure"})
  .on('click',(e) => P.resize(GPU_PARAMS.dimensions,uniform_scalar(0.0)));
f = pane.addFolder({title:"Engine"}) ;
  f.addInput(PARAMS.Engine,'StepsByFrame',{min:1, max:20, step:1})
  f.addInput(PARAMS.Engine,'PressureScale',{min:0.0, max:1.0})
f = pane.addFolder({title: "Material"}) ;
  f.addInput(PARAMS.Material,'painting')
  f.addInput(PARAMS.Material,'strength',{min:0,max:1})
  f.addInput(PARAMS.Material,'rho',{min:0.001, max:1.0})
  f.addInput(PARAMS.Material,'c',{min:0.001, max:1.0})
  f.addInput(PARAMS.Material,'eta_log',{min:-6, max:0})
  f.addButton({title: "Apply uniformly"})
  .on('click',(e) => {
    material_RGB.clearValue = mat2clr(PARAMS.Material);
    material_RGB.clear();
  });
f = pane.addFolder({title:"Rain"}) ;
  f.addInput(PARAMS.Rain,'Active')
  f.addInput(PARAMS.Rain,'DropDiameter',{min:1.0,max:100.0})
  f.addInput(PARAMS.Rain,'DropPeriod',{min:5,max:500})

// Init a composer.
const composer = new GPUComposer({canvas: canvas, 
                                  verboseLogging:false
                                });

// COMMON GPU LAYER PARAMETERS
GPU_PARAMS = {
  dimensions: [Math.floor(1*canvas.width/1), Math.floor(1*canvas.height/1)],
  type: FLOAT,
  filter: NEAREST,
  wrapX: REPEAT,
  wrapY: REPEAT,
}

// Load an image 
const img = new Image()
img.addEventListener('load', function() {
  //  exécute les instructions drawImage ici
  console.log(img)
}, false);
img.src = "img.png"

// Uniform initial arrays
function uniform_scalar(val) {return new Float32Array(GPU_PARAMS.dimensions[0] * GPU_PARAMS.dimensions[1]).fill(val);};
function uniform_vector(val) {return new Float32Array(2 * GPU_PARAMS.dimensions[0] * GPU_PARAMS.dimensions[1]).fill(val);};
function uniform_RGBA(val) {return new Float32Array(4 * GPU_PARAMS.dimensions[0] * GPU_PARAMS.dimensions[1]).fill(val);};

// Init the GPU Layers
// Pressure field
const P = new GPULayer(composer, Object.assign({},GPU_PARAMS,{
  name: 'pressure',
  numComponents: 1, // Scalar pressure field.
  numBuffers: 3, // 3 buffers for p(t), p(t-dt), p(t-2dt)
  array: uniform_scalar(0), // zero uniform pressure
}));
// Pressure gradient field
const gradP = new GPULayer(composer, Object.assign({},GPU_PARAMS,{
  name: 'pressure_gradient',
  numComponents: 2, // gradient vector in 2D.
  numBuffers: 1, // only one buffer needed
  array: uniform_vector(0), // zero uniform pressure gradient
}));
// Material RGB field
const material_RGB = new GPULayer(composer, Object.assign({},GPU_PARAMS,{
  name: 'material_RGB',
  numComponents: 4, // scalar field.
  numBuffers: 2, // allows updating by painting
  clearValue: mat2clr(PARAMS.Material),
}));
material_RGB.clear();
// Material Raw properties field
const material_Props = new GPULayer(composer, Object.assign({},GPU_PARAMS,{
  name: 'material_Props',
  numComponents: 4, // scalar field.
  numBuffers: 1, // allows updating by painting
  array: uniform_RGBA(1.0), // uniform intitial density
}));

// Pressure gradient computation program
const gradP_Program = new GPUProgram(composer, {
  name: 'gradP',
  fragmentShader: `
    in vec2 v_uv;

    uniform sampler2D u_pressure;
    uniform vec2 u_pxSize;

    out vec2 gradP;

    void main() {
      // discrete gradient with centered finite differences
      float n = texture(u_pressure, v_uv + vec2(0, u_pxSize.y)).x;
      float s = texture(u_pressure, v_uv - vec2(0, u_pxSize.y)).x;
      float e = texture(u_pressure, v_uv + vec2(u_pxSize.x, 0)).x;
      float w = texture(u_pressure, v_uv - vec2(u_pxSize.x, 0)).x;
      gradP = 0.5 * vec2(e-w,n-s);
    }
  `,
  uniforms: [
    { name: 'u_pressure', value: 0, type: INT, },
    { name: 'u_pxSize', value: [1 / GPU_PARAMS.dimensions[0], 1 / GPU_PARAMS.dimensions[1]], type: FLOAT, },
  ],
});

// RGB color to YUV material 
const RGBtoMatProgram = new GPUProgram(composer, {
  name: 'RGBtoMat',
  fragmentShader: `
    in vec2 v_uv;
    uniform sampler2D u_RGB;
    out vec4 out_mat;

    void main() {
      float R = texture(u_RGB,v_uv).x;
      float G = texture(u_RGB,v_uv).y;
      float B = texture(u_RGB,v_uv).z;
      float Y = (R + 2.0*G + B)/4.0; 
      float U = B - Y; 
      float V = R - Y;
      float rho = 1.0 - Y; 
      float c = U + 0.5; 
      float k = rho*c*c;
      float eta_log = 6.0*V - 3.0;
      out_mat = vec4(rho,k,eta_log,1.0);
    }
  `,
  uniforms: [
    { name: 'u_RGB', value: 0, type: INT, },
  ],
});

// Wave equation updating program
const waveProgram = new GPUProgram(composer, {
  name: 'wave',
  fragmentShader: `
    in vec2 v_uv;

    uniform sampler2D u_current_pressure; // p(t)
    uniform sampler2D u_previous_pressure; // p(t-dt)
    uniform sampler2D u_pressure_gradient; // grad(p(t))
    uniform sampler2D u_mat_props; // [rho,k,eta,1]
    uniform vec2 u_pxSize;

    out float next_p; // p(t+dt)

    void main() {
      float current_p = texture(u_current_pressure, v_uv).x;
      float previous_p = texture(u_previous_pressure, v_uv).x;
      float rho = texture(u_mat_props, v_uv).x;
      float eta = pow(10.0,texture(u_mat_props, v_uv).z);
      float n = texture(u_mat_props,v_uv + vec2(0, u_pxSize.y)).y * texture(u_pressure_gradient, v_uv + vec2(0, u_pxSize.y)).y;
      float s = texture(u_mat_props,v_uv - vec2(0, u_pxSize.y)).y * texture(u_pressure_gradient, v_uv - vec2(0, u_pxSize.y)).y;
      float e = texture(u_mat_props,v_uv + vec2(u_pxSize.x, 0)).y * texture(u_pressure_gradient, v_uv + vec2(u_pxSize.x, 0)).x;
      float w = texture(u_mat_props,v_uv - vec2(u_pxSize.x, 0)).y * texture(u_pressure_gradient, v_uv - vec2(u_pxSize.x, 0)).x;
      float divKgradP = 0.5 * (n - s + e - w);
      //next_p = (1.0-eta)*( (2.0 / rho * divKgradP) + (2.0 * current_p - previous_p) );

      float k = texture(u_mat_props, v_uv).y;
      float mu = -k*eta;
      float ap = divKgradP + (rho-mu)*current_p - (rho/2.0)*previous_p;
      next_p = ap/(rho/2.0-mu);
    }
  `,
  uniforms: [
    { name: 'u_current_pressure', value: 0, type: INT, },
    { name: 'u_previous_pressure', value: 1, type: INT, },
    { name: 'u_pressure_gradient', value: 2, type: INT, },
    { name: 'u_mat_props', value: 3, type: INT, },
    { name: 'u_pxSize', value: [1 / GPU_PARAMS.dimensions[0], 1 / GPU_PARAMS.dimensions[1]], type: FLOAT, },
  ],
});

// Drawing programs.
const hanningProgram = new GPUProgram(composer, {
  name: 'hanning',
  fragmentShader: `
    in vec2 v_uv;
    in vec2 v_uv_local;
    uniform sampler2D u_current_value;
    uniform float u_aspect_ratio;
    out float out_value;

    void main() {
      // hanning window
      vec2 vector = 2.0*v_uv_local-1.0;
      // The patch morphing produces a weird v_uv_local behavior
      if (abs(vector.x)>abs(vector.y)) {
        float Y = vector.y/vector.x;
        float xi = sqrt(1.0-Y*Y);
        vector.x = vector.x*(1.0 + xi/u_aspect_ratio);
      }
      // now x is the segment parameter
      float xe = max(0.0,(abs(vector.x)-1.0)*u_aspect_ratio) ; 
      float d = length(vec2(xe,vector.y)); // distance from the segment

      float co = cos(3.1416*d/2.0);
      out_value = texture(u_current_value,v_uv).x + co*co;
    }
  `,
  uniforms: [
    { name: 'u_current_value', value: 0, type: INT, },
    { name: 'u_aspect_ratio', value: 0.0, type: FLOAT, },
  ],
});
// Mouse pointer rendering
const addPointerProgram = new GPUProgram(composer, {
  name: 'pointer',
  fragmentShader: `
    in vec2 v_uv;
    in vec2 v_uv_local;
    uniform sampler2D u_current_rgba;
    uniform float u_aspect_ratio;
    out vec4 out_rgba;

    void main() {
      vec2 vector = 2.0*v_uv_local-1.0;
      // The patch morphing produces a weird v_uv_local behavior
      if (abs(vector.x)>abs(vector.y)) {
        float Y = vector.y/vector.x;
        float xi = sqrt(1.0-Y*Y);
        vector.x = vector.x*(1.0 + xi/u_aspect_ratio);
      }
      // now x is the segment parameter
      float xe = max(0.0,(abs(vector.x)-1.0)*u_aspect_ratio) ; 
      float d = length(vec2(xe,vector.y)); // distance from the segment

      //out_rgba = abs(d)*vec4((d>0.0),0.0,(d<0.0),0.0) + vec4(0.0,0.0,0.0,1);
      //out_rgba = ceil(out_rgba*6.0)/6.0;

      float e = .25;
      float fun = 1.0/e*float(d>1.0-2.0*e)*(e-abs(1.0-e-d));
      fun = sin(3.1426/2.0*fun);
      fun = fun*fun;

      out_rgba = texture(u_current_rgba,v_uv)*vec4(1.0,1.0,1.0,1.0);
      out_rgba = out_rgba - fun*vec4(1.0,1.0,1.0,0.0);
    }
  `,
  uniforms: [
    { name: 'u_current_rgba', value: 0, type: INT, },
    { name: 'u_aspect_ratio', value: 0.0, type: FLOAT, },
  ],
});
// Painter drawing.
const painterProgram = new GPUProgram(composer, {
  name: 'painter',
  fragmentShader: `
    in vec2 v_uv;
    in vec2 v_uv_local;
    uniform sampler2D u_current_rgba;
    uniform float u_aspect_ratio;
    uniform float u_strength;
    uniform vec4 u_paint_color;
    out vec4 out_rgba;

    void main() {
      // hanning window
      vec2 vector = 2.0*v_uv_local-1.0;
      // The patch morphing produces a weird v_uv_local behavior
      if (abs(vector.x)>abs(vector.y)) {
        float Y = vector.y/vector.x;
        float xi = sqrt(1.0-Y*Y);
        vector.x = vector.x*(1.0 + xi/u_aspect_ratio);
      }
      // now x is the segment parameter
      float xe = max(0.0,(abs(vector.x)-1.0)*u_aspect_ratio) ; 
      float d = length(vec2(xe,vector.y)); // distance from the segment
      
      float fun = 1.0;
      if (u_strength<1.0) {
        fun = (1.0-d); // normalizeddistance from the border [0,1]
        fun = fun/(1.0-u_strength); // strength 
        fun = min(fun,1.0); // top-hat function
        //fun = sin(3.1416/2.0*fun); // sinusoidal variation
        //fun = fun*fun; // hanning
      }
      out_rgba = texture(u_current_rgba,v_uv)*(1.0-fun) + fun*u_paint_color;
    }
  `,
  uniforms: [
    { name: 'u_current_rgba', value: 0, type: INT, },
    { name: 'u_aspect_ratio', value: 0.0, type: FLOAT, },
    { name: 'u_strength', value: PARAMS.Material.strength, type: FLOAT, },
    { name: 'u_paint_color', value: mat2clr(PARAMS.Material), type: FLOAT, },
  ],
});

// Add a segment to a given layer with a given program
function addSegment(layer,program,p1,p2,thickness) {
  // add some perturbation to the second point in order to force segment length
  p2 = math.add(p2,1e-9);
  // compute aspect ratio
  L = math.norm(math.subtract(p2,p1)) ;
  program.setUniform('u_aspect_ratio',L/thickness, FLOAT); // pass segment length to uniforms
  // Apply the program
  layer_bkp = layer.clone() ; // backup the layer to prevent buffer incrementation
  composer.stepSegment({
    program: program,
    position1: p1,
    position2: p2,
    thickness: thickness,
    endCaps: true,
    numCapSegments: 900,
    input: layer_bkp, // input with "layer" would have incremented its buffer
    output: layer,
    useOutputScale: true, // Use the same px scale size as the output GPULayer (otherwise it uses screen px).
    //blendAlpha: true, 
  });
  layer_bkp.dispose() ; // free memory
}

// RENDERING LAYERS & PROGRAMS
const drawLayer = new GPULayer(composer, Object.assign({},GPU_PARAMS,{
  name: 'draw',
  numComponents: 4, // R,G,B,A.
  numBuffers: 2, // allows updating
  clearValue: [.5,.5,.5,1.0],
  //array: uniform_RGBA(0.5), // zero uniform pressure
}));
// Drawing the pressure with signed amplitude
// const drawPressureProgram = renderSignedAmplitudeProgram(composer, {
//   name: 'drawPressure_SignedAmplitude',
//   type: GPU_PARAMS.type,
//   components: 'x',
// });
const drawPressureProgram = new GPUProgram(composer, {
  name: 'drawPressure_density',
  fragmentShader: `
    in vec2 v_uv;
    uniform sampler2D u_pressure;
    uniform sampler2D u_current_rgba;
    uniform float u_pressure_scaling;
    out vec4 out_rgba;

    void main() {
      out_rgba = texture(u_current_rgba,v_uv) + texture(u_pressure,v_uv).x*vec4(1.0,1.0,1.0,0.0)*u_pressure_scaling;
    }
  `,
  uniforms: [
    { name: 'u_pressure', value: 0, type: INT, },
    { name: 'u_current_rgba', value: 1, type: INT, },
    { name: 'u_pressure_scaling', value: PARAMS.Engine.PressureScale, type: FLOAT, },
  ],
});
// Final rendering
const renderProgram = copyProgram(composer, {
  name: 'render',
  type: GPU_PARAMS.type,
});


// Define mouse interactions
const UI = {
  activebuttons: [],
  mouseInCanvas: false,
  pointerStart: [0,0],
  pointerPosition: [0,0],
  pointerDiameter: 10,
}
function callback(e) {
  e.preventDefault(); 
  if (e.type=='contextmenu') ; 
  if (e.type=='pointermove') {
    UI.pointerPosition=[e.clientX,canvas.height-e.clientY];
    if (e.buttons==0) UI.pointerStart = UI.pointerPosition
  }
  if (e.type=='pointerover') UI.mouseInCanvas=true;
  if (e.type=='pointerout') UI.mouseInCanvas=false;
  if (e.type=='wheel') UI.pointerDiameter=Math.max(3,Math.min(UI.pointerDiameter*1.5**Math.sign(e.wheelDelta),300));
  if (e.type=='pointerdown') UI.pointerStart = UI.pointerPosition;
  if (e.type=='pointerup') {
    if (PARAMS.Material.painting) addSegment(material_RGB,painterProgram,UI.pointerStart,UI.pointerPosition,UI.pointerDiameter);
    else addSegment(P,hanningProgram,UI.pointerStart,UI.pointerPosition,UI.pointerDiameter);
  }
  //if (e.type=='pointerdown' || e.type=='pointerup') console.log(e.button);
  // console.log(e.type);
  // console.log(UI.pointerPosition);
}
[
  'pointerdown',
  'pointerup',
  'pointermove',
  //'click',
  'pointerover',
  'pointerout',
  'pointercancel',
  'wheel',
  'contextmenu',
].forEach((evt,i)=>canvas.addEventListener(evt, callback));


let nextDropTime = performance.now();

// Simulation/render loop.
function loop() {
  fpsGraph.begin();

  // ADD DROP ?
  if (PARAMS.Rain.Active & performance.now()>=nextDropTime) {
    position = [Math.random()*GPU_PARAMS.dimensions[0],Math.random()*GPU_PARAMS.dimensions[1]]
    diameter = PARAMS.Rain.DropDiameter.min + (PARAMS.Rain.DropDiameter.max-PARAMS.Rain.DropDiameter.min)*Math.random();
    addSegment(P,hanningProgram,position,position,diameter);
    period = PARAMS.Rain.DropPeriod.min + (PARAMS.Rain.DropPeriod.max-PARAMS.Rain.DropPeriod.min)*Math.random();
    nextDropTime = performance.now() + period;
    // console.log("Drop!")
  }
  
  // VARIABLE UPDATING
  for (let it=0; it<PARAMS.Engine.StepsByFrame ; it++) {
    // Compute the material properties from the RGB layer
    composer.step({
      program: RGBtoMatProgram,
      input: material_RGB,
      output: material_Props,
    });
    // Compute the pressure gradient
    composer.step({
      program: gradP_Program,
      input: P,
      output: gradP,
    });
    // Apply the wave equation.
    composer.step({
      program: waveProgram,
      input: [
            P.currentState,
            P.lastState,
            gradP,
            material_Props
          ],
      output: P,
    });
  }


  // RENDERING

  // Draw the material on background
  composer.step({
    program: renderProgram,
    input: material_RGB,
    output: drawLayer,
  });

  // Draw the pressure
  drawPressureProgram.setUniform('u_pressure_scaling',PARAMS.Engine.PressureScale)
  composer.step({
    program: drawPressureProgram,
    input: [P.currentState,drawLayer],
    output: drawLayer,
  });


  // ADD THE POINTER
  if (UI.mouseInCanvas) {
    if (PARAMS.Material.painting) {
      painterProgram.setUniform('u_strength',PARAMS.Material.strength)
      painterProgram.setUniform('u_paint_color',mat2clr(PARAMS.Material))
      addSegment(drawLayer,painterProgram,UI.pointerStart,UI.pointerPosition,UI.pointerDiameter);
    }
    else addSegment(drawLayer,addPointerProgram,UI.pointerStart,UI.pointerPosition,UI.pointerDiameter);
  }

  // If no "output", will draw to canvas.
  composer.step({
    program: renderProgram,
    input: drawLayer,
  });


  fpsGraph.end();
  window.requestAnimationFrame(loop);
}
loop(); // Start animation loop.