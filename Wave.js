
const {
    GPUComposer,
    GPULayer,
    GPUProgram,
    renderAmplitudeProgram,
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
  SubSteps: 1, // number of stepping substeps
  WaveVelocity: .5, // wave velocity (pix/step)
  Damping: .001, // wave velocity
  DropRadius: 5, // source radius in pixels
  DropPeriod: 200, // drop period in millisecs
  Sensor: {
    Position: { x:0.0, y:0.0 }, 
    Output: 0, 
  },
  theme: 'dark',
}

// Init a simple gui. (see https://cocopon.github.io/tweakpane/quick-tour/)
const pane = new Tweakpane.Pane();
pane.addInput(PARAMS,'WaveVelocity',{min:0.0, max:.5})
  .on('change',(e) => waveProgram.setUniform('u_waveVelocity', e.value));
pane.addInput(PARAMS,'Damping',{min:0.0, max:1.0})
  .on('change',(e) => waveProgram.setUniform('u_damping', e.value));
pane.addInput(PARAMS,'SubSteps',{min:1, max:100, step:1})
pane.addInput(PARAMS,'DropRadius',{min:1.0, max:100.0})
pane.addInput(PARAMS,'DropPeriod',{min:10.0, max:2000.0})
sensor = pane.addFolder({title:'Sensor'});
sensor.addInput(PARAMS.Sensor,'Position',{picker:'inline',x:{min:-1.0,max:1.0},y:{min:-1.0,max:1.0}});
sensor.addMonitor(PARAMS.Sensor, 'Output', {view: 'graph',min:-1,max:1});

// Init a composer.
const composer = new GPUComposer({canvas, 
                                  verboseLogging:false
                                });

// Initial state
const init_state = new Float32Array(canvas.width * canvas.height); 
// for (let i = 0; i < init_state.length; i++) {
//   let y = Math.floor(i/canvas.width);
//   let x = Math.floor(i - (y*canvas.width));
//   // init_state[i] = Math.random(); // random noise
//   // init_state[i] = x/canvas.width; // x axis
//   // init_state[i] = y/canvas.height; // y axis
//   let x0 = canvas.width/2; let y0 = canvas.height/2; let R2 = PARAMS.ropRadius**2;
//   let r2 = (x-x0)**2 + (y-y0)**2;
//   // init_state[i] = r2 < R2 ; // uniform disk
//   // init_state[i] = 1*Math.max(0,(R2-r2)/R2) ; // hat
//   // init_state[i] = 1*Math.exp(-r2/R2) ; // gaussian
// }

// Init the layer of float data.
const state = new GPULayer(composer, {
  name: 'state',
  dimensions: [canvas.width, canvas.height],
  numComponents: 1, // Scalar state has one component.
  type: FLOAT,
  filter: NEAREST,
  // Use 3 buffers : u(t), u(t-dt), u(t-2*dt)
  numBuffers: 3,
  wrapX: CLAMP_TO_EDGE,
  wrapY: CLAMP_TO_EDGE,
  array: init_state
}); 

// Wave equation updating program
const waveProgram = new GPUProgram(composer, {
  name: 'render',
  fragmentShader: `
    in vec2 v_uv;

    uniform sampler2D u_current_state;
    uniform sampler2D u_previous_state;
    uniform vec2 u_pxSize;
    uniform float u_waveVelocity;
    uniform float u_damping;

    out float out_result;

    void main() {
      // Compute the discrete Laplacian.
      // https://en.wikipedia.org/wiki/Discrete_Laplace_operator
      float current = texture(u_current_state, v_uv).x;
      float previous = texture(u_previous_state, v_uv).x;
      float n = texture(u_current_state, v_uv + vec2(0, u_pxSize.y)).x;
      float s = texture(u_current_state, v_uv - vec2(0, u_pxSize.y)).x;
      float e = texture(u_current_state, v_uv + vec2(u_pxSize.x, 0)).x;
      float w = texture(u_current_state, v_uv - vec2(u_pxSize.x, 0)).x;
      float laplacian = n + s + e + w - 4.0 * current;
      out_result = (1.0-u_damping)*(2.0*u_waveVelocity*u_waveVelocity * laplacian + 2.0 * current - previous);
    }
  `,
  uniforms: [
    { // Index of sampler2D uniform to assign to value "u_state".
      name: 'u_current_state',
      value: 0,
      type: INT,
    },
    { // Index of sampler2D uniform to assign to value "u_state".
      name: 'u_previous_state',
      value: 1,
      type: INT,
    },
    { // Calculate the size of a 1 px step in UV coordinates.
      name: 'u_pxSize',
      value: [1 / canvas.width, 1 / canvas.height],
      type: FLOAT,
    },
    { // WaveVelocity as parameter
      name: 'u_waveVelocity',
      value: PARAMS.WaveVelocity,
      type: FLOAT,
    },
    { // WaveVelocity as parameter
      name: 'u_damping',
      value: PARAMS.Damping,
      type: FLOAT,
    },
  ],
});

// Drop program.
const dropProgram = new GPUProgram(composer, {
  name: 'drop',
  fragmentShader: `
    // We get v_uv_local when calling programs via stepCircle().
    // It gives the uv coordinates in the local reference frame of the circle.
    // See https://github.com/amandaghassaei/gpu-io/blob/main/docs/GLSL.md#fragment-shader-inputs
    in vec2 v_uv_local;
    out float out_height;
    void main() {
      // Calculate height so that it's tallest in the center and
      // tapers down toward the outside of the circle.
      // Use dist from center (vec2(0.5)) to compute this.
      vec2 vector = v_uv_local - vec2(0.5);
      float co = cos(3.1416*length(vector));
      out_height = co*co;
      // out_height = 1.0 - 2.0 * length(vector);
    }
  `,
});

function addDrop(position,radius) {
  // We need to be sure to write drop height to both lastState and currentState.
  // Write drop to lastState.
  // state.decrementBufferIndex();
  // composer.stepCircle({
  //   program: dropProgram,
  //   position,
  //   diameter: radius*2,
  //   output: state,
  //   useOutputScale: true, // Use the same px scale size as the output GPULayer (otherwise it uses screen px).
  // });
  // // Write drop to currentState.
  // state.incrementBufferIndex();
  composer.stepCircle({
    program: dropProgram,
    position,
    diameter: radius*2,
    output: state,
    useOutputScale: true, // Use the same px scale size as the output GPULayer (otherwise it uses screen px).
  });
}

// Init a program to render state to canvas.
// See https://github.com/amandaghassaei/gpu-io/tree/main/docs#gpuprogram-helper-functions
// for more built-in GPUPrograms to use.
const renderProgram = renderAmplitudeProgram(composer, {
  name: 'render',
  type: state.type,
  components: 'x',
});

let lastDropTime = performance.now();

// Simulation/render loop.
function loop() {

  // ADD DROP ?
  if (performance.now()-lastDropTime>PARAMS.DropPeriod) {
    position = [Math.random()*canvas.width,Math.random()*canvas.height]
    addDrop(position,PARAMS.DropRadius);
    lastDropTime = performance.now();
  }
  
  // VARIABLE UPDATING
  for (let it=0; it<PARAMS.SubSteps ; it++) {
    // Diffuse state and write result to state.
    composer.step({
      program: waveProgram,
      input: [state.currentState,state.lastState],
      output: state,
    });
  }

  // READ THE SENSOR VALUE
  let val = new Float32Array(1);
  let gl = composer.gl;
  let x = Math.floor(.5*(PARAMS.Sensor.Position.x+1.0)*canvas.width);
  let y = Math.floor(.5*(PARAMS.Sensor.Position.y+1.0)*canvas.height);
  gl.readPixels(x,y,1,1,gl.RED,gl.FLOAT,val)
  PARAMS.Sensor.Output = val[0];


  // RENDERING

  // If no "output", will draw to canvas.
  composer.step({
    program: renderProgram,
    input: state,
  });

  // Update fps counter.
  const { fps, numTicks } = composer.tick();
  if (numTicks % 10 === 0) {
    pane.title = `2D Wave (${fps.toFixed(1)} FPS)`;
  }
  pane.refresh();

  window.requestAnimationFrame(loop);
}
loop(); // Start animation loop.