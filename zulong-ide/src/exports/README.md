# Zulong API

The Zulong extension exposes an API that can be used by other extensions. To use this API in your extension:

1. Copy `src/extension-api/zulong.d.ts` to your extension's source directory.
2. Include `zulong.d.ts` in your extension's compilation.
3. Get access to the API with the following code:

    ```ts
    const zulongExtension = vscode.extensions.getExtension<ZulongAPI>("zulong.zulong")

    if (!zulongExtension?.isActive) {
    	throw new Error("Zulong extension is not activated")
    }

    const zulong = zulongExtension.exports

    if (zulong) {
    	// Now you can use the API

    	// Start a new task with an initial message
    	await zulong.startNewTask("Hello, Zulong! Let's make a new project...")

    	// Start a new task with an initial message and images
    	await zulong.startNewTask("Use this design language", ["data:image/webp;base64,..."])

    	// Send a message to the current task
    	await zulong.sendMessage("Can you fix the @problems?")

    	// Simulate pressing the primary button in the chat interface (e.g. 'Save' or 'Proceed While Running')
    	await zulong.pressPrimaryButton()

    	// Simulate pressing the secondary button in the chat interface (e.g. 'Reject')
    	await zulong.pressSecondaryButton()
    } else {
    	console.error("Zulong API is not available")
    }
    ```

    **Note:** To ensure that the `saoudrizwan.claude-dev` extension is activated before your extension, add it to the `extensionDependencies` in your `package.json`:

    ```json
    "extensionDependencies": [
        "saoudrizwan.claude-dev"
    ]
    ```

For detailed information on the available methods and their usage, refer to the `zulong.d.ts` file.
