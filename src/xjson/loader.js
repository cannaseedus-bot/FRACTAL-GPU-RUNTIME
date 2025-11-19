/**
 * XJSON Model Loader - loads transformer models from XJSON format
 * Compatible with @xjson/xjson-server format
 */

export class XJSONLoader {
    constructor() {
        this.models = new Map();
    }

    /**
     * Load model from XJSON file or URL
     */
    async load(source) {
        let modelData;

        if (typeof source === 'string') {
            if (source.startsWith('http')) {
                // Load from URL
                const response = await fetch(source);
                modelData = await response.json();
            } else {
                // Load from file (Node.js)
                const { readFileSync } = await import('fs');
                modelData = JSON.parse(readFileSync(source, 'utf-8'));
            }
        } else {
            modelData = source; // Already parsed
        }

        return this.parse(modelData);
    }

    /**
     * Parse XJSON model structure
     */
    parse(data) {
        const model = {
            config: data.config || {},
            architecture: data.architecture || data.model?.architecture || 'transformer',
            layers: [],
            weights: new Map(),
            metadata: data.metadata || {}
        };

        // Parse configuration
        model.config = {
            vocab_size: data.config?.vocab_size || 50257,
            n_embd: data.config?.n_embd || data.config?.hidden_size || 768,
            n_head: data.config?.n_head || data.config?.num_attention_heads || 12,
            n_layer: data.config?.n_layer || data.config?.num_hidden_layers || 12,
            max_seq_len: data.config?.max_seq_len || data.config?.max_position_embeddings || 1024,
            ...data.config
        };

        // Parse layers
        if (data.layers) {
            model.layers = data.layers.map(layer => this.parseLayer(layer));
        } else if (data.graph) {
            // Parse from graph representation
            model.layers = this.parseGraph(data.graph);
        }

        // Parse weights
        if (data.weights) {
            for (const [name, weight] of Object.entries(data.weights)) {
                model.weights.set(name, this.parseWeight(weight));
            }
        }

        this.models.set(model.metadata.name || 'model', model);
        return model;
    }

    parseLayer(layerData) {
        return {
            type: layerData.type || layerData.op,
            name: layerData.name || layerData.id,
            params: layerData.params || layerData.attributes || {},
            inputs: layerData.inputs || [],
            outputs: layerData.outputs || []
        };
    }

    parseGraph(graphData) {
        const layers = [];

        // Convert graph nodes to layers
        if (graphData.nodes) {
            for (const node of graphData.nodes) {
                layers.push({
                    type: node.op || node.type,
                    name: node.name || node.id,
                    params: node.attrs || {},
                    inputs: node.inputs || [],
                    outputs: node.outputs || []
                });
            }
        }

        return layers;
    }

    parseWeight(weightData) {
        if (weightData instanceof Float32Array) {
            return weightData;
        }

        if (Array.isArray(weightData)) {
            return new Float32Array(weightData.flat(Infinity));
        }

        if (weightData.data) {
            // Handle structured weight format
            const data = Array.isArray(weightData.data)
                ? new Float32Array(weightData.data.flat(Infinity))
                : weightData.data;

            return {
                data: data,
                shape: weightData.shape || [],
                dtype: weightData.dtype || 'float32'
            };
        }

        return weightData;
    }

    /**
     * Create a minimal transformer model structure
     */
    static createMiniTransformer(config = {}) {
        const defaultConfig = {
            vocab_size: 1000,
            n_embd: 128,
            n_head: 4,
            n_layer: 2,
            max_seq_len: 256,
        };

        const finalConfig = { ...defaultConfig, ...config };

        return {
            config: finalConfig,
            architecture: 'transformer',
            layers: [
                {
                    type: 'embedding',
                    name: 'token_embedding',
                    params: {
                        vocab_size: finalConfig.vocab_size,
                        embedding_dim: finalConfig.n_embd
                    }
                },
                ...Array.from({ length: finalConfig.n_layer }, (_, i) => ({
                    type: 'transformer_block',
                    name: `layer_${i}`,
                    params: {
                        n_embd: finalConfig.n_embd,
                        n_head: finalConfig.n_head
                    }
                })),
                {
                    type: 'layer_norm',
                    name: 'final_norm',
                    params: { n_embd: finalConfig.n_embd }
                },
                {
                    type: 'linear',
                    name: 'lm_head',
                    params: {
                        in_features: finalConfig.n_embd,
                        out_features: finalConfig.vocab_size
                    }
                }
            ],
            weights: new Map(),
            metadata: {
                name: 'mini_transformer',
                version: '0.1.0',
                created: new Date().toISOString()
            }
        };
    }
}
