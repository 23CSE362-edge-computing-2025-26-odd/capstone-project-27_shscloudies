import machine
import time
import urequests
import ujson
import network

# Configuration
DEVICE_ID = "pico_01"
SEND_INTERVAL = 5 
NODE_RED_URL = "https://webhook.site/6d9f4fd6-51f9-4323-a21d-8b908a6fa400"
…            print("\nStopping...")
            break
        except Exception as e:
            print(f"Error in main loop: {e}")
            time.sleep(SEND_INTERVAL)

if __name__ == "__main__":
    main()
