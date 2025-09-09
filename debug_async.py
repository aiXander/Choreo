#!/usr/bin/env python3
"""
Minimal test script to debug the async batch processing issue.
"""

import asyncio
import json
from typing import List, Dict, Any, Optional
from litellm import aresponses

class SimpleAsyncTester:
    """Minimal version of the async batch processing logic."""
    
    def __init__(self):
        self.call_count = 0
    
    async def batch_json_complete(
        self,
        prompts: List[str],
        model: str,
        batch_size: int = 16
    ) -> List[Dict[str, Any]]:
        """Simplified version of batch_json_complete."""
        if not prompts:
            return []
        
        n_prompts = len(prompts)
        print(f"Processing {n_prompts} prompts in batches of {batch_size}")
        
        results = [None] * n_prompts
        
        # Process prompts in batches
        for batch_start in range(0, n_prompts, batch_size):
            batch_end = min(batch_start + batch_size, n_prompts)
            batch_indices = list(range(batch_start, batch_end))
            
            print(f"Processing batch {batch_start//batch_size + 1}: items {batch_start + 1}-{batch_end} of {n_prompts}")
            
            # Create tasks for this batch
            tasks = []
            for i in batch_indices:
                prompt = prompts[i]
                task = asyncio.create_task(self._async_single_complete(prompt, model))
                tasks.append(task)
            
            print(f"Created {len(tasks)} tasks, now gathering...")
            
            # Execute batch
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            print(f"Gather completed, got {len(batch_results)} results")
            
            # Store results
            for i, result in zip(batch_indices, batch_results):
                if isinstance(result, Exception):
                    print(f"Task failed for prompt {i}: {result}")
                    results[i] = {"error": str(result)}
                else:
                    results[i] = result
            
            print(f"Stored results for batch {batch_start//batch_size + 1}")
        
        print(f"Completed processing all {n_prompts} prompts")
        return results
    
    async def _async_single_complete(
        self,
        prompt: str,
        model: str
    ) -> Dict[str, Any]:
        """Single async completion."""
        print(f"Starting async call for prompt: {prompt[:50]}...")
        
        try:
            # Make async LLM call via Responses API
            response = await aresponses(
                input=[{ 'role': 'developer', 'content': prompt }],
                model=model,
                stream=False,
                background=False,
                reasoning = {
                    "effort": "low"
                },
            )
            
            print(f"Got response for prompt: {prompt}...")
            print(response.output[1].content[0].text)
            
            # Extract content from Responses API response
            content: Optional[str] = None


            content = response.output[1].content[0].text

            if not content:
                raise ValueError("Empty response from LLM")
            
            # Try to parse as JSON, fallback to text
            try:
                result = json.loads(content.strip())
                print(f"Parsed JSON successfully for prompt: {prompt[:30]}...")
                return result
            except json.JSONDecodeError as e:
                print(f"Could not parse JSON, returning text for prompt: {prompt[:30]}...")
                print(content.strip())
                print(f"Error: {e}")
                return {"response": content.strip()}
                
        except Exception as e:
            print(f"Exception in _async_single_complete: {e}")
            raise

async def async_main():
    """Async version of main to properly handle cleanup."""
    print("=== Async Batch Test Started ===")
    
    # Create simple test prompts
    test_prompts = [
        "Return a JSON object with a 'test' field set to 1. Reply with only the json, nothing else.",
        "Return a JSON object with a 'test' field set to 2. Reply with only the json, nothing else.", 
        "Return a JSON object with a 'test' field set to 3. Reply with only the json, nothing else.",
        "Return a JSON object with a 'test' field set to 4. Reply with only the json, nothing else."
    ]
    
    model = "gpt-5-mini"  # Use a fast, cheap model
    
    print("About to call batch_json_complete...")
    
    try:
        tester = SimpleAsyncTester()
        results = await tester.batch_json_complete(
            prompts=test_prompts,
            model=model,
            batch_size=16
        )
        
        print("=== BATCH COMPLETED SUCCESSFULLY ===")
        print(f"Got {len(results)} results:")
        for i, result in enumerate(results):
            print(f"  Result {i}: {result}")
        
    except Exception as e:
        print(f"=== ERROR IN BATCH PROCESSING ===")
        print(f"Exception: {e}")
        import traceback
        traceback.print_exc()
    
    print("=== Test completed ===")
    await cleanup_background_tasks()

async def cleanup_background_tasks():
    """Cancel any lingering background tasks to ensure clean exit."""
    pending = asyncio.all_tasks()
    current = asyncio.current_task()
    background_tasks = [task for task in pending if task != current and not task.done()]
    
    if background_tasks:
        print(f"Cleaning up {len(background_tasks)} background tasks")
        for task in background_tasks:
            task.cancel()
        await asyncio.sleep(0.1)  # Give tasks time to cancel

def main():
    """Test the async batch processing with proper cleanup."""
    asyncio.run(async_main())

if __name__ == "__main__":
    main()