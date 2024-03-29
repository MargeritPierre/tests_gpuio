
const {
    GPUComposer,
    GPULayer,
    GPUProgram,
    renderAmplitudeProgram,
    FLOAT,
    INT,
    REPEAT,
    NEAREST,
  } = GPUIO ;
  
  // Init a canvas element.
  const canvas = document.createElement('canvas');
  document.body.appendChild(canvas);
  console.log(canvas);
  
  // Init a composer.
  const composer = new GPUComposer({ canvas });
  
  // Init a layer of float data filled with noise.
  const noise = new Float32Array(canvas.width * canvas.height);
  noise.forEach((el, i) => noise[i] = Math.random());
  const state = new GPULayer(composer, {
    name: 'state',
    dimensions: [canvas.width, canvas.height],
    numComponents: 1, // Scalar state has one component.
    type: FLOAT,
    filter: NEAREST,
    // Use 2 buffers so we can toggle read/write
    // from one to the other.
    numBuffers: 2,
    wrapX: REPEAT,
    wrapY: REPEAT,
    array: noise,
  });
  
  // Init a program to diffuse state.
  const diffuseProgram = new GPUProgram(composer, {
    name: 'render',
    fragmentShader: `
      in vec2 v_uv;
  
      uniform sampler2D u_state;
      uniform vec2 u_pxSize;
  
      out float out_result;
  
      void main() {
        // Compute the discrete Laplacian.
        // https://en.wikipedia.org/wiki/Discrete_Laplace_operator
        float center = texture(u_state, v_uv).x;
        float n = texture(u_state, v_uv + vec2(0, u_pxSize.y)).x;
        float s = texture(u_state, v_uv - vec2(0, u_pxSize.y)).x;
        float e = texture(u_state, v_uv + vec2(u_pxSize.x, 0)).x;
        float w = texture(u_state, v_uv - vec2(u_pxSize.x, 0)).x;
        const float diffusionRate = 0.1;
        out_result =
          center + diffusionRate * (n + s + e + w - 4.0 * center);
      }
    `,
    uniforms: [
      { // Index of sampler2D uniform to assign to value "u_state".
        name: 'u_state',
        value: 0,
        type: INT,
      },
      { // Calculate the size of a 1 px step in UV coordinates.
        name: 'u_pxSize',
        value: [1 / canvas.width, 1 / canvas.height],
        type: FLOAT,
      },
    ],
  });
  
  // Init a program to render state to canvas.
  // See https://github.com/amandaghassaei/gpu-io/tree/main/docs#gpuprogram-helper-functions
  // for more built-in GPUPrograms to use.
  const renderProgram = renderAmplitudeProgram(composer, {
    name: 'render',
    type: state.type,
    components: 'x',
  });
  
  // Simulation/render loop.
  function loop() {
    window.requestAnimationFrame(loop);
  
    // Diffuse state and write result to state.
    composer.step({
      program: diffuseProgram,
      input: state,
      output: state,
    });
  
    // If no "output", will draw to canvas.
    composer.step({
      program: renderProgram,
      input: state,
    });
  }
  loop(); // Start animation loop.