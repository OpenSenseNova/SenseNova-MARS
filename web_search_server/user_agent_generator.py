# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Tuple


class UserAgentGenerator:
    """Generates realistic, diverse user agents to avoid bot detection"""
    
    @staticmethod
    def generate_realistic_user_agents(pool_size: int) -> List[str]:
        """Generate diverse, realistic user agents to avoid bot detection"""
        user_agents = []
        
        # Chrome versions (recent and realistic)
        chrome_versions = [
            "122.0.6261.57", "121.0.6167.85", "120.0.6099.109", "119.0.6045.105",
            "118.0.5993.70", "117.0.5938.92", "116.0.5845.96", "115.0.5790.102"
        ]
        
        # Firefox versions
        firefox_versions = [
            "123.0", "122.0", "121.0", "120.0", "119.0", "118.0", "117.0", "116.0"
        ]
        
        # Safari versions (WebKit)
        safari_versions = [
            "537.36", "605.1.15", "604.1", "603.3.8", "602.4.8"
        ]
        
        # Operating systems with realistic distributions
        operating_systems = [
            # Windows (most common)
            ("Windows NT 10.0; Win64; x64", 0.7),
            ("Windows NT 11.0; Win64; x64", 0.15),
            # macOS
            ("Macintosh; Intel Mac OS X 10_15_7", 0.08),
            ("Macintosh; Intel Mac OS X 11_7_10", 0.03),
            ("Macintosh; Intel Mac OS X 12_6_3", 0.02),
            # Linux
            ("X11; Linux x86_64", 0.015),
            ("X11; Ubuntu; Linux x86_64", 0.005)
        ]
        
        # Browser types with market share
        browsers = [
            ("chrome", 0.65),
            ("edge", 0.13),
            ("firefox", 0.12),
            ("safari", 0.08),
            ("opera", 0.02)
        ]
        
        for i in range(pool_size):
            # Use deterministic randomization based on index for consistency
            seed_base = hash(f"ua_seed_{i}") % 1000000
            
            # Select browser type
            browser_rand = (seed_base * 17) % 1000 / 1000.0
            browser_type = "chrome"  # default
            cumulative = 0
            for browser, prob in browsers:
                cumulative += prob
                if browser_rand <= cumulative:
                    browser_type = browser
                    break
            
            # Select OS
            os_rand = (seed_base * 23) % 1000 / 1000.0
            os_string = operating_systems[0][0]  # default
            cumulative = 0
            for os_info, prob in operating_systems:
                cumulative += prob
                if os_rand <= cumulative:
                    os_string = os_info
                    break
            
            # Generate user agent based on browser type
            user_agent = UserAgentGenerator._generate_user_agent_for_browser(
                browser_type, os_string, seed_base, chrome_versions, 
                firefox_versions, safari_versions
            )
            
            user_agents.append(user_agent)
        
        return user_agents
    
    @staticmethod
    def _generate_user_agent_for_browser(browser_type: str, os_string: str, seed_base: int,
                                       chrome_versions: List[str], firefox_versions: List[str], 
                                       safari_versions: List[str]) -> str:
        """Generate user agent string for specific browser type"""
        
        if browser_type == "chrome":
            version = chrome_versions[(seed_base * 31) % len(chrome_versions)]
            webkit_version = safari_versions[(seed_base * 37) % len(safari_versions)]
            return f"Mozilla/5.0 ({os_string}) AppleWebKit/{webkit_version} (KHTML, like Gecko) Chrome/{version} Safari/{webkit_version}"
            
        elif browser_type == "edge":
            version = chrome_versions[(seed_base * 41) % len(chrome_versions)]
            webkit_version = safari_versions[(seed_base * 43) % len(safari_versions)]
            edge_version = f"{version.split('.')[0]}.{(seed_base * 47) % 9999}.{(seed_base * 53) % 999}.{(seed_base * 59) % 99}"
            return f"Mozilla/5.0 ({os_string}) AppleWebKit/{webkit_version} (KHTML, like Gecko) Chrome/{version} Safari/{webkit_version} Edg/{edge_version}"
            
        elif browser_type == "firefox":
            version = firefox_versions[(seed_base * 61) % len(firefox_versions)]
            gecko_version = f"{20100101 + (seed_base * 67) % 999999}"
            if "Windows" in os_string:
                return f"Mozilla/5.0 ({os_string}; rv:{version}) Gecko/{gecko_version} Firefox/{version}"
            elif "Macintosh" in os_string:
                return f"Mozilla/5.0 ({os_string}) Gecko/{gecko_version} Firefox/{version}"
            else:  # Linux
                return f"Mozilla/5.0 ({os_string}; rv:{version}) Gecko/{gecko_version} Firefox/{version}"
                
        elif browser_type == "safari":
            if "Macintosh" not in os_string:
                # Fallback to Chrome for non-Mac
                version = chrome_versions[(seed_base * 71) % len(chrome_versions)]
                webkit_version = safari_versions[(seed_base * 73) % len(safari_versions)]
                return f"Mozilla/5.0 ({os_string}) AppleWebKit/{webkit_version} (KHTML, like Gecko) Chrome/{version} Safari/{webkit_version}"
            else:
                webkit_version = f"605.{1 + (seed_base * 79) % 5}.{15 + (seed_base * 83) % 10}"
                safari_version = f"{17 + (seed_base * 89) % 3}.{(seed_base * 97) % 10}"
                return f"Mozilla/5.0 ({os_string}) AppleWebKit/{webkit_version} (KHTML, like Gecko) Version/{safari_version} Safari/{webkit_version}"
                
        else:  # opera
            chrome_version = chrome_versions[(seed_base * 101) % len(chrome_versions)]
            opera_version = f"{106 + (seed_base * 103) % 10}.{(seed_base * 107) % 9}.{(seed_base * 109) % 9999}.{(seed_base * 113) % 999}"
            webkit_version = safari_versions[(seed_base * 127) % len(safari_versions)]
            return f"Mozilla/5.0 ({os_string}) AppleWebKit/{webkit_version} (KHTML, like Gecko) Chrome/{chrome_version} Safari/{webkit_version} OPR/{opera_version}"