import { app } from "../../../scripts/app.js";

function replaceNode(oldNode, newNodeName) {
    const newNode = LiteGraph.createNode(newNodeName);
    if (!newNode) {
        return;
    }
    app.graph.add(newNode);

    newNode.pos = oldNode.pos.slice();

    // Handle the special nodes with two outputs
    const nodesWithTwoOutputs = ["XY Input: LoRA Plot", "XY Input: Control Net Plot", "XY Input: Manual XY Entry"];
    let outputCount = nodesWithTwoOutputs.includes(oldNode.type) ? 2 : 1;

    // Transfer output connections from old node to new node
    oldNode.outputs.slice(0, outputCount).forEach((output, index) => {
        if (output && output.links) {
            output.links.forEach(link => {
                const targetLinkInfo = oldNode.graph.links[link];
                if (targetLinkInfo) {
                    const targetNode = oldNode.graph.getNodeById(targetLinkInfo.target_id);
                    if (targetNode) {
                        newNode.connect(index, targetNode, targetLinkInfo.target_slot);
                    }
                }
            });
        }
    });

    // Remove old node
    app.graph.remove(oldNode);
}

const xyInputNodes = [
    "XY Input: Seeds++ Batch",
    "XY Input: Add/Return Noise",
    "XY Input: Steps",
    "XY Input: CFG Scale",
    "XY Input: Sampler/Scheduler",
    "XY Input: Denoise",
    "XY Input: VAE",
    "XY Input: Prompt S/R",
    "XY Input: Aesthetic Score",
    "XY Input: Refiner On/Off",
    "XY Input: Checkpoint",
    "XY Input: Clip Skip",
    "XY Input: LoRA",
    "XY Input: LoRA Plot",
    "XY Input: LoRA Stacks",
    "XY Input: Control Net",
    "XY Input: Control Net Plot",
    "XY Input: Manual XY Entry"
];

// Extension Definition
app.registerExtension({
    name: "efficiency.swapXYinputs",
    getNodeMenuItems(node) {
        if (!node.comfyClass?.startsWith("XY Input:")) return [];

        const options = xyInputNodes
            .filter(n => n !== node.comfyClass)
            .map(n => ({ content: n, callback: () => replaceNode(node, n) }));

        return [{ content: "🔄 Swap with...", submenu: { options } }];
    },
});
